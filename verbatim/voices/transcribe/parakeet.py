"""NeMo Parakeet transcription with word-level timestamps."""

import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from verbatim.model_cache import is_local_model_reference, resolve_hf_file_path
from verbatim_transcript import TranscriberCapabilities

from ...transcript.words import Word
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)
SAMPLE_RATE = 16000


def _model_path(model_size_or_path: str) -> str:
    if is_local_model_reference(model_size_or_path):
        return os.path.expanduser(model_size_or_path)
    return resolve_hf_file_path(
        model_size_or_path,
        filename=f"{model_size_or_path.rsplit('/', 1)[-1]}.nemo",
        purpose="Parakeet model",
    )


def prefetch_parakeet_model(model_size_or_path: str) -> None:
    """Download the NeMo checkpoint used by the selected Parakeet model."""
    _model_path(model_size_or_path)


class ParakeetTranscriber(Transcriber):
    """Transcribe mono 16 kHz audio using a configurable Parakeet checkpoint."""

    CAPABILITIES = TranscriberCapabilities(
        accepts_language_hint=False,
        detects_language=False,
        reports_language=False,
        supports_prompt=False,
        supports_prefix=False,
    )

    def __init__(self, *, model_size_or_path: str, device: str):
        try:
            from nemo.collections.asr.models import ASRModel  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("Parakeet requires NeMo. Install `verbatim[nemo]`.") from exc

        model_path = _model_path(model_size_or_path)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Parakeet model must be a .nemo checkpoint: {model_path}")

        self._model: Any = ASRModel.restore_from(restore_path=model_path, map_location=device)
        self._model.eval()
        self._model.to(device)

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        _ = audio, lang
        raise RuntimeError("Parakeet does not expose language identification; select a separate language identifier.")

    def transcribe(
        self,
        *,
        audio: NDArray,
        lang: Optional[str],
        prompt: str,
        prefix: str,
        window_ts: int,
        audio_ts: int,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Optional[List[float]] = None,
    ) -> List[Word]:
        _ = prompt, prefix, whisper_beam_size, whisper_best_of, whisper_patience, whisper_temperatures
        if lang is not None:
            raise RuntimeError("Parakeet does not accept language hints.")
        if audio.ndim != 1:
            raise ValueError("Parakeet requires mono audio.")
        if audio.size == 0:
            return []

        signal = np.ascontiguousarray(audio, dtype=np.float32)
        hypothesis = self._model.transcribe([signal], batch_size=1, timestamps=True)[0]
        timestamps = hypothesis.timestamp.get("word", [])
        words: List[Word] = []
        for stamp in timestamps:
            token = str(stamp["word"])
            if not token:
                continue
            start_ts = window_ts + round(float(stamp["start"]) * SAMPLE_RATE)
            end_ts = window_ts + round(float(stamp["end"]) * SAMPLE_RATE)
            start_ts = max(window_ts, min(start_ts, audio_ts))
            end_ts = max(start_ts, min(end_ts, audio_ts))
            if end_ts == start_ts:
                continue
            if words and not token.startswith((" ", "'", "’", ".", ",", "!", "?", ":", ";")):
                token = " " + token
            words.append(Word(start_ts=start_ts, end_ts=end_ts, word=token, probability=1.0, lang="und"))
        LOG.debug("Parakeet returned %d words for samples %d-%d", len(words), window_ts, audio_ts)
        return words
