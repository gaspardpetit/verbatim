import contextlib
import logging
import os
import sys
import time
from typing import Any, Optional

# pylint: disable=import-outside-toplevel,broad-exception-caught
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from torch.serialization import add_safe_globals

from verbatim.cache import ArtifactCache
from verbatim.logging_utils import get_status_logger, status_enabled
from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_diarization.pyannote.separate import PyannoteSpeakerSeparation, _build_rttm_annotation
from verbatim_diarization.pyannote_annotations import to_rttm_annotation
from verbatim_diarization.utils import sanitize_uri_component
from verbatim_files.rttm import Annotation as RTTMAnnotation
from verbatim_files.vttm import AudioRef, dumps_vttm

from .constants import PYANNOTE_DIARIZATION_MODEL_ID
from .ffmpeg_loader import ensure_torchcodec_audio_decoder

LOG = logging.getLogger(__name__)
STATUS_LOG = get_status_logger()


class PyAnnoteDiarization(DiarizationStrategy):
    def __init__(self, *, cache: ArtifactCache, device: str, huggingface_token: str):
        super().__init__(cache=cache)
        self.device = device
        self.huggingface_token = huggingface_token
        self.pipeline = None
        self.model_id = PYANNOTE_DIARIZATION_MODEL_ID  # previously "pyannote/speaker-diarization-3.1"

    def initialize_pipeline(self):
        """Lazy initialization of PyAnnote pipeline"""
        if self.pipeline is None:
            # Allow safe loading of pyannote checkpoints under torch>=2.6
            torch_version_mod: Any = getattr(torch, "torch_version", None)
            safe_types = [Specifications, Problem, Resolution]
            if torch_version_mod is not None:
                safe_types.append(torch_version_mod.TorchVersion)  # type: ignore[attr-defined]
            add_safe_globals(safe_types)  # pyright: ignore[reportArgumentType]
            # Default to community-friendly model
            self.pipeline = Pipeline.from_pretrained(self.model_id, token=self.huggingface_token)
        if self.pipeline is None:
            raise RuntimeError("PyAnnote pipeline failed to initialize")
        self.pipeline.instantiate({})
        self.pipeline.to(torch.device(self.device))

    # pylint: disable=too-many-positional-arguments
    def compute_diarization(
        self,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_vttm_file: Optional[str] = None,
        nb_speakers: Optional[int] = None,
        **kwargs,
    ) -> RTTMAnnotation:
        """
        Compute diarization using PyAnnote.

        Additional kwargs:
            nb_speakers: Optional number of speakers
        """
        cache_bytes = self.cache.get_bytes(file_path)
        cache_len = len(cache_bytes)
        LOG.debug(
            "pyannote input probe: path=%s cache_bytes=%s",
            file_path,
            cache_len,
        )
        # pyannote.audio 4.x requires torchcodec; ensure it is ready before pipeline work.
        ensure_torchcodec_audio_decoder("pyannote diarization")
        diarization = None
        try:
            try:
                import soundfile as sf  # lazy import

                buffer = self.cache.bytes_io(file_path)
                audio, sample_rate = sf.read(buffer)
                LOG.debug("pyannote cache decode ok: path=%s sr=%s shape=%s", file_path, sample_rate, getattr(audio, "shape", None))

                if isinstance(audio, np.ndarray) and audio.ndim > 1 and audio.shape[1] > 1:
                    mono = np.mean(audio, axis=1)
                else:
                    mono = audio

                waveform = torch.from_numpy(np.asarray(mono, dtype=np.float32))
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                file_for_pipeline = {"waveform": waveform, "sample_rate": int(sample_rate)}
            except Exception as exc:
                raise RuntimeError(f"PyAnnote diarization requires cached audio bytes; failed to decode cache for '{file_path}': {exc}") from exc

            self.initialize_pipeline()
            pipeline = self.pipeline
            if pipeline is None:
                raise RuntimeError("PyAnnote pipeline failed to initialize")
            start = time.perf_counter()
            show_progress = status_enabled()
            if show_progress:
                with contextlib.redirect_stdout(sys.stderr):
                    with ProgressHook() as hook:
                        diarization = pipeline(file_for_pipeline, hook=hook, num_speakers=nb_speakers)
            else:
                diarization = pipeline(file_for_pipeline, num_speakers=nb_speakers)
            elapsed = time.perf_counter() - start
            STATUS_LOG.info(
                "PyAnnote diarization completed in %.2fs (model=%s, file=%s, speakers=%s)",
                elapsed,
                self.model_id,
                file_path,
                nb_speakers if nb_speakers is not None else "auto",
            )
        finally:
            pass

        if diarization is None:
            raise RuntimeError("PyAnnote diarization failed without producing output")

        # pyannote.audio 4.x returns a DiarizeOutput with a speaker_diarization field
        if hasattr(diarization, "speaker_diarization"):
            diarization_annotation = diarization.speaker_diarization  # type: ignore[attr-defined]
        else:
            diarization_annotation = diarization

        raw_uri = getattr(diarization_annotation, "uri", None) or os.path.splitext(os.path.basename(file_path))[0]
        uri = sanitize_uri_component(str(raw_uri))
        try:
            diarization_annotation.uri = uri  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive: Annotation may freeze attributes
            LOG.debug("Unable to set sanitized URI on diarization annotation", exc_info=True)

        rttm_annotation = to_rttm_annotation(diarization_annotation)

        if out_rttm_file:
            self.save_rttm(rttm_annotation, out_rttm_file)

        if out_vttm_file:
            self.cache.set_text(
                out_vttm_file,
                dumps_vttm(audio=[AudioRef(id=uri, path=file_path, channels=None)], annotation=rttm_annotation),
            )

        return rttm_annotation


class PyAnnoteSeparationDiarization(DiarizationStrategy):
    def __init__(self, *, cache: ArtifactCache, device: str, huggingface_token: str):
        super().__init__(cache=cache)
        self.device = device
        self.huggingface_token = huggingface_token
        self._separator = PyannoteSpeakerSeparation(cache=cache, device=device, huggingface_token=huggingface_token)

    # pylint: disable=too-many-positional-arguments
    def compute_diarization(
        self,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_vttm_file: Optional[str] = None,
        nb_speakers: Optional[int] = None,
        stem_prefix: Optional[str] = None,
        **kwargs,
    ) -> RTTMAnnotation:
        del kwargs
        uri = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0])
        base_dir = os.path.dirname(stem_prefix or "")
        if not stem_prefix:
            stem_prefix = os.path.join(base_dir or ".", f"{uri}-speaker")

        diarization, audio_refs_meta = self._separator._separate_to_audio_refs(  # pylint: disable=protected-access
            file_path=file_path,
            out_speaker_wav_prefix=stem_prefix,
            nb_speakers=nb_speakers,
        )
        label_to_ref = dict(audio_refs_meta)
        annotation = _build_rttm_annotation(diarization, label_to_ref, uri)

        if out_rttm_file:
            self.save_rttm(annotation, out_rttm_file)

        if out_vttm_file:
            audio_refs = [ref for _label, ref in audio_refs_meta]
            self.cache.set_text(out_vttm_file, dumps_vttm(audio=audio_refs, annotation=annotation))

        return annotation
