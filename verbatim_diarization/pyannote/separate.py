import logging
from typing import Any, List, Optional, cast

import numpy as np
import scipy.io.wavfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from torch.serialization import add_safe_globals

from verbatim.audio.audio import wav_to_int16
from verbatim.audio.sources.audiosource import AudioSource
from verbatim.audio.sources.fileaudiosource import FileAudioSource
from verbatim_diarization.diarize.factory import create_diarizer
from verbatim_diarization.separate.base import SeparationStrategy

from .constants import PYANNOTE_SEPARATION_MODEL_ID
from .ffmpeg_loader import ensure_ffmpeg_for_torchcodec

# Configure logger
LOG = logging.getLogger(__name__)


class PyannoteSpeakerSeparation(SeparationStrategy):
    def __init__(
        self,
        device: str,
        huggingface_token: str,
        separation_model=PYANNOTE_SEPARATION_MODEL_ID,
        diarization_strategy: str = "pyannote",
        **kwargs,
    ):
        super().__init__()
        LOG.info("Initializing Separation Pipeline.")
        self.diarization_strategy = diarization_strategy
        self.device = device
        self.huggingface_token = huggingface_token
        # Allow safe loading of pyannote checkpoints under torch>=2.6
        torch_version_mod: Any = getattr(torch, "torch_version", None)
        safe_types = [Specifications, Problem, Resolution]
        if torch_version_mod is not None:
            safe_types.append(torch_version_mod.TorchVersion)  # type: ignore[attr-defined]
        add_safe_globals(safe_types)  # pyright: ignore[reportArgumentType]

        self.pipeline = Pipeline.from_pretrained(separation_model, token=self.huggingface_token)
        hyper_parameters = {
            "segmentation": {"min_duration_off": 0.0, "threshold": 0.82},
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": 0.68,
            },
            "separation": {
                "leakage_removal": True,
                "asr_collar": 0.32,
            },
        }

        if self.pipeline is None:
            raise RuntimeError("Pyannote separation pipeline failed to initialize")

        self.pipeline.instantiate(hyper_parameters)
        self.pipeline.to(torch.device(device))

    def __enter__(self) -> "PyannoteSpeakerSeparation":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        del self.pipeline
        return False

    # pylint: disable=unused-argument
    def separate_speakers(
        self,
        *,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_vttm_file: Optional[str] = None,
        out_speaker_wav_prefix="",
        nb_speakers: Optional[int] = None,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
    ) -> List[AudioSource]:
        """
        Separate speakers in an audio file.

        Args:
            file_path: Path to input audio file
            out_rttm_file: Optional legacy RTTM output path
            out_speaker_wav_prefix: Prefix for output WAV files
            nb_speakers: Optional number of speakers

        Returns:
            Tuple of (diarization annotation, dictionary mapping speaker IDs to WAV files)
        """
        separated_sources: List[AudioSource] = []
        audio_refs_meta: List[tuple[str, str]] = []
        diarization_annotation = None
        if not out_rttm_file:
            out_rttm_file = None

        # For stereo strategy, we might want to handle separation differently
        if self.diarization_strategy == "stereo":
            # For stereo files, we can simply split the channels
            sample_rate, audio_data = scipy.io.wavfile.read(file_path)
            if audio_data.ndim != 2 or audio_data.shape[1] != 2:  # type: ignore[index]
                raise ValueError("Stereo separation requires stereo audio input")

            # Create diarization annotation
            diarizer = create_diarizer(strategy="stereo", device=self.device, huggingface_token=self.huggingface_token)
            diarization = diarizer.compute_diarization(
                file_path=file_path, out_rttm_file=out_rttm_file, out_vttm_file=out_vttm_file, nb_speakers=nb_speakers
            )

            # Split channels into separate files
            for channel, speaker in enumerate(["SPEAKER_0", "SPEAKER_1"]):
                channel_data = audio_data[:, channel]
                if channel_data.dtype != np.int16:
                    channel_data = wav_to_int16(channel_data)
                file_name = f"{out_speaker_wav_prefix}-{speaker}.wav" if out_speaker_wav_prefix else f"{speaker}.wav"
                scipy.io.wavfile.write(file_name, sample_rate, channel_data)
                audio_refs_meta.append((speaker, file_name))
                separated_sources.append(
                    FileAudioSource(
                        file=file_name,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        diarization=diarization,
                    )
                )
            diarization_annotation = diarization

        else:
            # Use PyAnnote's neural separation for mono files
            ensure_ffmpeg_for_torchcodec()

            def _import_torchcodec():
                from torchcodec.decoders import AudioDecoder  # noqa: F401

                try:
                    import pyannote.audio.core.io as pa_io

                    setattr(pa_io, "AudioDecoder", AudioDecoder)  # pyright: ignore[reportPrivateImportUsage]
                except Exception:
                    pass

            try:
                _import_torchcodec()
            except Exception:
                ensure_ffmpeg_for_torchcodec()
                try:
                    _import_torchcodec()
                except Exception as exc2:  # pragma: no cover - defensive import
                    raise RuntimeError(
                        "pyannote separation could not load torchcodec (FFmpeg dependency). "
                        "Install FFmpeg shared libraries (4–7) and set FFMPEG_DLL_DIR or add them to PATH."
                    ) from exc2
            with ProgressHook() as hook:
                if self.pipeline is None:
                    raise RuntimeError("Pyannote separation pipeline is not initialized")
                diarization_output, sources = self.pipeline(file_path, hook=hook, num_speakers=nb_speakers)

            # pyannote.audio 4.x returns DiarizeOutput; normalize to Annotation
            diarization = diarization_output.speaker_diarization if hasattr(diarization_output, "speaker_diarization") else diarization_output

            # Save diarization to RTTM file if requested
            if out_rttm_file:
                with open(out_rttm_file, "w", encoding="utf-8") as rttm:
                    diarization.write_rttm(rttm)
            diarization_annotation = diarization

            # Save separated sources to WAV files
            sources_data = np.asarray(sources.data)
            shape = cast(tuple[int, int], tuple(sources_data.shape[:2]))

            for s, speaker in enumerate(diarization.labels()):
                if s < shape[1]:
                    speaker_data = sources_data[:, s]
                    if speaker_data.dtype != np.int16:
                        speaker_data = wav_to_int16(speaker_data)
                    file_name = f"{out_speaker_wav_prefix}-{speaker}.wav" if out_speaker_wav_prefix else f"{speaker}.wav"
                    scipy.io.wavfile.write(file_name, 16000, speaker_data)
                    audio_refs_meta.append((speaker, file_name))
                    separated_sources.append(
                        FileAudioSource(
                            file=file_name,
                            start_sample=start_sample,
                            end_sample=end_sample,
                            diarization=diarization,
                        )
                    )
                else:
                    LOG.debug(f"Skipping speaker {s} as it is out of bounds.")

        if out_vttm_file and diarization_annotation is not None:
            import os

            from verbatim_rttm import Annotation as RTTMAnnotation
            from verbatim_rttm import AudioRef, Segment, write_vttm

            uri = os.path.splitext(os.path.basename(file_path))[0]
            label_to_path = {label: path for label, path in audio_refs_meta}
            segments = []
            for segment, _track, label in diarization_annotation.itertracks(yield_label=True):
                seg_label = str(label)
                file_id = seg_label if seg_label in label_to_path else uri
                segments.append(Segment(start=segment.start, end=segment.end, speaker=seg_label, file_id=file_id))

            audio_refs = [AudioRef(id=label, path=path, channel="1") for label, path in audio_refs_meta] or [AudioRef(id=uri, path=file_path)]

            write_vttm(out_vttm_file, audio=audio_refs, annotation=RTTMAnnotation(segments=segments, file_id=uri))

        return separated_sources
