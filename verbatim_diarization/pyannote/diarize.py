import io
import logging
import os
import time
from typing import Any, Optional

# pylint: disable=import-outside-toplevel,broad-exception-caught
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation
from torch.serialization import add_safe_globals

from verbatim.cache import get_default_cache
from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_diarization.pyannote.separate import PyannoteSpeakerSeparation, _build_rttm_annotation
from verbatim_diarization.utils import sanitize_uri_component
from verbatim_files.rttm import Annotation as RTTMAnnotation
from verbatim_files.rttm import Segment
from verbatim_files.vttm import AudioRef, write_vttm

from .constants import PYANNOTE_DIARIZATION_MODEL_ID
from .ffmpeg_loader import ensure_torchcodec_audio_decoder

LOG = logging.getLogger(__name__)


class PyAnnoteDiarization(DiarizationStrategy):
    def __init__(self, device: str, huggingface_token: str):
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
        working_dir: Optional[str] = None,
        **kwargs,
    ) -> Annotation:
        """
        Compute diarization using PyAnnote.

        Additional kwargs:
            nb_speakers: Optional number of speakers
        """
        # pyannote.audio 4.x requires torchcodec; ensure it is ready before pipeline work.
        ensure_torchcodec_audio_decoder("pyannote diarization")
        diarization = None
        try:
            try:
                import soundfile as sf  # lazy import

                if os.path.exists(file_path):
                    audio, sample_rate = sf.read(file_path)
                else:
                    cache = get_default_cache()
                    cached = cache.get_bytes(file_path) if cache else None
                    if cached is None:
                        raise FileNotFoundError(f"Audio file not found: {file_path}")
                    audio, sample_rate = sf.read(io.BytesIO(cached))

                if isinstance(audio, np.ndarray) and audio.ndim > 1 and audio.shape[1] > 1:
                    mono = np.mean(audio, axis=1)
                else:
                    mono = audio

                waveform = torch.from_numpy(np.asarray(mono, dtype=np.float32))
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                file_for_pipeline = {"waveform": waveform, "sample_rate": int(sample_rate)}
            except Exception:
                # Fallback: let pyannote handle; may fail if multi-channel unsupported
                file_for_pipeline = file_path

            self.initialize_pipeline()
            pipeline = self.pipeline
            if pipeline is None:
                raise RuntimeError("PyAnnote pipeline failed to initialize")
            start = time.perf_counter()
            with ProgressHook() as hook:
                diarization = pipeline(file_for_pipeline, hook=hook, num_speakers=nb_speakers)
            elapsed = time.perf_counter() - start
            LOG.info(
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

        if out_rttm_file:
            self.save_rttm(diarization_annotation, out_rttm_file)

        if out_vttm_file:
            os.makedirs(os.path.dirname(out_vttm_file) or ".", exist_ok=True)
            segments = [
                Segment(start=segment.start, end=segment.end, speaker=str(label), file_id=uri)
                for segment, _track, label in diarization_annotation.itertracks(yield_label=True)
            ]
            rttm_ann = RTTMAnnotation(segments=segments, file_id=uri)
            write_vttm(out_vttm_file, audio=[AudioRef(id=uri, path=file_path, channels=None)], annotation=rttm_ann)

        return diarization_annotation


class PyAnnoteSeparationDiarization(DiarizationStrategy):
    def __init__(self, device: str, huggingface_token: str):
        self.device = device
        self.huggingface_token = huggingface_token
        self._separator = PyannoteSpeakerSeparation(device=device, huggingface_token=huggingface_token)

    # pylint: disable=too-many-positional-arguments
    def compute_diarization(
        self,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_vttm_file: Optional[str] = None,
        nb_speakers: Optional[int] = None,
        working_dir: Optional[str] = None,
        stem_prefix: Optional[str] = None,
        **kwargs,
    ) -> RTTMAnnotation:
        del kwargs
        uri = sanitize_uri_component(os.path.splitext(os.path.basename(file_path))[0])
        base_dir = working_dir or os.path.dirname(stem_prefix or "") or tempfile.gettempdir()
        if not stem_prefix:
            stem_prefix = os.path.join(base_dir or ".", f"{uri}-speaker")

        diarization, audio_refs_meta = self._separator._separate_to_audio_refs(  # pylint: disable=protected-access
            file_path=file_path,
            out_speaker_wav_prefix=stem_prefix,
            nb_speakers=nb_speakers,
            working_dir=working_dir,
        )
        label_to_ref = dict(audio_refs_meta)
        annotation = _build_rttm_annotation(diarization, label_to_ref, uri)

        if out_rttm_file:
            self.save_rttm(annotation, out_rttm_file)

        if out_vttm_file:
            os.makedirs(os.path.dirname(out_vttm_file) or ".", exist_ok=True)
            audio_refs = [ref for _label, ref in audio_refs_meta]
            write_vttm(out_vttm_file, audio=audio_refs, annotation=annotation)

        return annotation
