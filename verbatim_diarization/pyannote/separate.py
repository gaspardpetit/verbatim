import contextlib
import io
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

# pylint: disable=import-outside-toplevel,broad-exception-caught
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from torch.serialization import add_safe_globals

from verbatim.cache import ArtifactCache
from verbatim.logging_utils import get_status_logger, status_enabled
from verbatim_audio.audio import wav_to_int16
from verbatim_audio.sources.audiosource import AudioSource
from verbatim_audio.sources.fileaudiosource import FileAudioSource
from verbatim_diarization.separate.base import SeparationStrategy
from verbatim_diarization.utils import sanitize_uri_component
from verbatim_files.rttm import Annotation as RTTMAnnotation
from verbatim_files.rttm import Segment, dumps_rttm
from verbatim_files.vttm import AudioRef, dumps_vttm

from .constants import PYANNOTE_SEPARATION_MODEL_ID
from .ffmpeg_loader import ensure_torchcodec_audio_decoder

LOG = logging.getLogger(__name__)
STATUS_LOG = get_status_logger()


def _build_rttm_annotation(diarization, label_to_ref: Dict[str, AudioRef], default_uri: str) -> RTTMAnnotation:
    segments = []
    for segment, _track, label in diarization.itertracks(yield_label=True):
        seg_label = str(label)
        audio_ref = label_to_ref.get(seg_label)
        file_id = audio_ref.id if audio_ref else default_uri
        segments.append(Segment(start=segment.start, end=segment.end, speaker=seg_label, file_id=file_id))
    return RTTMAnnotation(segments=segments, file_id=default_uri)


class PyannoteSpeakerSeparation(SeparationStrategy):
    def __init__(
        self,
        *,
        cache: ArtifactCache,
        device: str,
        huggingface_token: str,
        separation_model=PYANNOTE_SEPARATION_MODEL_ID,
        diarization_strategy: str = "pyannote",
        **kwargs,
    ):
        super().__init__(cache=cache)
        del kwargs  # unused
        STATUS_LOG.info("Initializing Separation Pipeline.")
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

    def _prepare_pipeline_input(self, file_path: str) -> Tuple[Any, Optional[str]]:
        temp_path: Optional[str] = None
        try:
            import soundfile as sf  # lazy import

            buffer = self.cache.bytes_io(file_path)
            audio, sample_rate = sf.read(buffer)

            if isinstance(audio, np.ndarray) and audio.ndim > 1 and audio.shape[1] > 1:
                mono = np.mean(audio, axis=1)
            else:
                mono = audio

            waveform = torch.from_numpy(np.asarray(mono, dtype=np.float32))
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            return {"waveform": waveform, "sample_rate": int(sample_rate)}, None
        except Exception:
            temp_path = None
        return file_path, temp_path

    def _separate_to_audio_refs(
        self,
        *,
        file_path: str,
        out_speaker_wav_prefix: str,
        nb_speakers: Optional[int],
    ) -> Tuple[Any, List[Tuple[str, AudioRef]]]:
        ensure_torchcodec_audio_decoder("pyannote separation")
        file_for_pipeline, temp_path = self._prepare_pipeline_input(file_path)
        try:
            if self.pipeline is None:
                raise RuntimeError("Pyannote separation pipeline is not initialized")
            show_progress = status_enabled()
            if show_progress:
                with contextlib.redirect_stdout(sys.stderr):
                    with ProgressHook() as hook:
                        diarization_output, sources = self.pipeline(file_for_pipeline, hook=hook, num_speakers=nb_speakers)
            else:
                diarization_output, sources = self.pipeline(file_for_pipeline, num_speakers=nb_speakers)
        finally:
            if temp_path and temp_path != file_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:  # pragma: no cover
                    LOG.debug("Failed to clean up temporary separation file %s", temp_path)

        diarization = diarization_output.speaker_diarization if hasattr(diarization_output, "speaker_diarization") else diarization_output

        sources_data = np.asarray(sources.data)
        shape = cast(tuple[int, int], tuple(sources_data.shape[:2]))

        audio_refs_meta: List[Tuple[str, AudioRef]] = []
        for s, speaker in enumerate(diarization.labels()):
            if s >= shape[1]:
                LOG.debug("Skipping speaker index %s as it is out of bounds for separated sources", s)
                continue

            speaker_data = sources_data[:, s]
            if speaker_data.dtype != np.int16:
                speaker_data = wav_to_int16(speaker_data)

            file_name = f"{out_speaker_wav_prefix}-{speaker}.wav" if out_speaker_wav_prefix else f"{speaker}.wav"
            import soundfile as sf  # lazy import

            buffer = io.BytesIO()
            sf.write(buffer, speaker_data, 16000, format="WAV")
            self.cache.set_bytes(file_name, buffer.getvalue())

            sanitized_id = sanitize_uri_component(os.path.splitext(os.path.basename(file_name))[0])
            audio_ref = AudioRef(id=sanitized_id, path=file_name, channels=None)
            audio_refs_meta.append((str(speaker), audio_ref))

        if not audio_refs_meta:
            uri = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0])
            audio_refs_meta.append((uri, AudioRef(id=uri, path=file_path, channels=None)))

        return diarization, audio_refs_meta

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
            List of FileAudioSource entries, one per separated speaker.
        """
        diarization, audio_refs_meta = self._separate_to_audio_refs(
            file_path=file_path,
            out_speaker_wav_prefix=out_speaker_wav_prefix,
            nb_speakers=nb_speakers,
        )

        uri = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0])
        label_to_ref = dict(audio_refs_meta)
        diarization_annotation = _build_rttm_annotation(diarization, label_to_ref, uri)

        separated_sources: List[AudioSource] = []
        for _label, audio_ref in audio_refs_meta:
            separated_sources.append(
                FileAudioSource(
                    file=audio_ref.path,
                    cache=self.cache,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    diarization=diarization_annotation,
                    file_id=audio_ref.id,
                )
            )

        if out_rttm_file:
            self.cache.set_text(out_rttm_file, dumps_rttm(diarization_annotation))

        if out_vttm_file:
            audio_refs = [ref for _label, ref in audio_refs_meta]
            self.cache.set_text(out_vttm_file, dumps_vttm(audio=audio_refs, annotation=diarization_annotation))

        return separated_sources
