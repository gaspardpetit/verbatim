import logging
import os
from typing import Any, Dict, List, Optional, cast

# pylint: disable=import-outside-toplevel,broad-exception-caught
import numpy as np
import scipy.io.wavfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from torch.serialization import add_safe_globals

from verbatim_audio.audio import wav_to_int16
from verbatim_audio.sources.audiosource import AudioSource
from verbatim_audio.sources.fileaudiosource import FileAudioSource
from verbatim_diarization.separate.base import SeparationStrategy
from verbatim_files.rttm import Annotation as RTTMAnnotation
from verbatim_files.rttm import Segment, write_rttm
from verbatim_files.vttm import AudioRef, write_vttm

from .constants import PYANNOTE_SEPARATION_MODEL_ID
from .ffmpeg_loader import ensure_torchcodec_audio_decoder

# Configure logger
LOG = logging.getLogger(__name__)


def _build_rttm_annotation(diarization, label_to_path: Dict[str, str], default_uri: str) -> RTTMAnnotation:
    segments = []
    for segment, _track, label in diarization.itertracks(yield_label=True):
        seg_label = str(label)
        file_id = seg_label if seg_label in label_to_path else default_uri
        segments.append(Segment(start=segment.start, end=segment.end, speaker=seg_label, file_id=file_id))
    return RTTMAnnotation(segments=segments, file_id=default_uri)


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
        del kwargs  # unused
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

        # Use PyAnnote's neural separation for mono/mixed audio
        ensure_torchcodec_audio_decoder("pyannote separation")
        with ProgressHook() as hook:
            if self.pipeline is None:
                raise RuntimeError("Pyannote separation pipeline is not initialized")
            diarization_output, sources = self.pipeline(file_path, hook=hook, num_speakers=nb_speakers)

        # pyannote.audio 4.x returns DiarizeOutput; normalize to Annotation
        diarization = diarization_output.speaker_diarization if hasattr(diarization_output, "speaker_diarization") else diarization_output

        # Save diarization to RTTM file if requested
        uri = os.path.splitext(os.path.basename(file_path))[0]

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

        label_to_path = dict(audio_refs_meta)
        cached_annotation: Optional[RTTMAnnotation] = None

        def ensure_rttm_annotation() -> RTTMAnnotation:
            nonlocal cached_annotation
            if cached_annotation is None:
                cached_annotation = _build_rttm_annotation(diarization_annotation, label_to_path, uri)
            return cached_annotation

        if out_rttm_file:
            write_rttm(ensure_rttm_annotation(), out_rttm_file)

        if out_vttm_file:
            audio_refs = [AudioRef(id=label, path=path, channels="1") for label, path in audio_refs_meta] or [
                AudioRef(id=uri, path=file_path, channels="1")
            ]
            write_vttm(out_vttm_file, audio=audio_refs, annotation=ensure_rttm_annotation())

        return separated_sources
