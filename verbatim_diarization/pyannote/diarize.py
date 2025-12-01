import logging
import os
import time
from typing import Optional

import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation
from torch.serialization import add_safe_globals

from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_rttm import Annotation as RTTMAnnotation
from verbatim_rttm import AudioRef, Segment, write_vttm

from .ffmpeg_loader import ensure_ffmpeg_for_torchcodec

LOG = logging.getLogger(__name__)


class PyAnnoteDiarization(DiarizationStrategy):
    def __init__(self, device: str, huggingface_token: str):
        self.device = device
        self.huggingface_token = huggingface_token
        self.pipeline = None
        self.model_id = "pyannote/speaker-diarization-community-1"  # previously "pyannote/speaker-diarization-3.1"

    def initialize_pipeline(self):
        """Lazy initialization of PyAnnote pipeline"""
        if self.pipeline is None:
            # Allow safe loading of pyannote checkpoints under torch>=2.6
            add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution])
            # Default to community-friendly model
            self.pipeline = Pipeline.from_pretrained(self.model_id, token=self.huggingface_token)
        if self.pipeline is None:
            raise RuntimeError("PyAnnote pipeline failed to initialize")
        self.pipeline.instantiate({})
        self.pipeline.to(torch.device(self.device))

    def compute_diarization(
        self, file_path: str, out_rttm_file: Optional[str] = None, out_vttm_file: Optional[str] = None, nb_speakers: Optional[int] = None, **kwargs
    ) -> Annotation:
        """
        Compute diarization using PyAnnote.

        Additional kwargs:
            nb_speakers: Optional number of speakers
        """
        # pyannote.audio 4.x requires torchcodec; attempt to ensure FFmpeg DLLs first.
        ensure_ffmpeg_for_torchcodec()

        def _import_torchcodec():
            from torchcodec.decoders import AudioDecoder  # noqa: F401

            try:
                import pyannote.audio.core.io as pa_io

                pa_io.AudioDecoder = AudioDecoder
            except Exception:
                pass

        try:
            _import_torchcodec()
        except Exception as exc:
            # Retry after another FFmpeg scan; surface a clearer error if still broken.
            ensure_ffmpeg_for_torchcodec()
            try:
                _import_torchcodec()
            except Exception as exc2:  # pragma: no cover - defensive import
                raise RuntimeError(
                    "pyannote diarization could not load torchcodec (FFmpeg dependency). "
                    "Install FFmpeg shared libraries (4–7) and set FFMPEG_DLL_DIR or add them to PATH."
                ) from exc2
        self.initialize_pipeline()
        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("PyAnnote pipeline failed to initialize")
        start = time.perf_counter()
        with ProgressHook() as hook:
            diarization = pipeline(file_path, hook=hook, num_speakers=nb_speakers)
        elapsed = time.perf_counter() - start
        LOG.info(
            "PyAnnote diarization completed in %.2fs (model=%s, file=%s, speakers=%s)",
            elapsed,
            self.model_id,
            file_path,
            nb_speakers if nb_speakers is not None else "auto",
        )

        # pyannote.audio 4.x returns a DiarizeOutput with a speaker_diarization field
        if hasattr(diarization, "speaker_diarization"):
            diarization_annotation = diarization.speaker_diarization  # type: ignore[attr-defined]
        else:
            diarization_annotation = diarization

        if out_rttm_file:
            self.save_rttm(diarization_annotation, out_rttm_file)

        if out_vttm_file:
            os.makedirs(os.path.dirname(out_vttm_file) or ".", exist_ok=True)
            uri = os.path.splitext(os.path.basename(file_path))[0]
            segments = [
                Segment(start=segment.start, end=segment.end, speaker=str(label), file_id=uri)
                for segment, _track, label in diarization_annotation.itertracks(yield_label=True)
            ]
            rttm_ann = RTTMAnnotation(segments=segments, file_id=uri)
            write_vttm(out_vttm_file, audio=[AudioRef(id=uri, path=file_path)], annotation=rttm_ann)

        return diarization_annotation
