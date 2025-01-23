import logging
from abc import ABC, abstractmethod
from typing import Dict, List
import warnings

import numpy as np
from numpy.typing import NDArray
import torch

from silero_vad import get_speech_timestamps, load_silero_vad

# Configure logger
LOG = logging.getLogger(__name__)


class VoiceActivityDetection(ABC):
    @abstractmethod
    def find_activity(
        self,
        audio: NDArray,
        threshold: float = 0.25,
        neg_threshold: float | None = None,
        min_silence_duration_ms: int = 100,
        min_speech_duration_ms: int = 250,
        speech_pad_ms: int = 250,
    ) -> List[Dict[str, int]]:
        pass


class SileroVoiceActivityDetection(VoiceActivityDetection):
    def __init__(self):
        with warnings.catch_warnings():
            # suppresses silero_vad\model.py:15: DeprecationWarning: path is deprecated. Use files() instead.
            # Refer to https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy for migration advice.
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.model = load_silero_vad()

    def _normalize_audio(self, audio: NDArray) -> NDArray:
        # Normalize audio to range [-1, 1]
        if len(audio) == 0:
            return audio
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            return audio / max_abs
        return audio

    def find_activity(
        self,
        audio: NDArray,
        threshold: float = 0.25,
        neg_threshold: float | None = None,
        min_silence_duration_ms: int = 100,
        min_speech_duration_ms: int = 250,
        speech_pad_ms: int = 250,
    ) -> List[Dict[str, int]]:
        # Normalize audio first
        audio = self._normalize_audio(audio)
        audio_tensor = torch.from_numpy(audio).float()

        speech_timestamps = get_speech_timestamps(
            audio=audio_tensor,
            model=self.model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,
            visualize_probs=False,
            progress_tracking_callback=None,  # pyright: ignore[reportArgumentType]
            # When neg_threshold = None, then neg_threshold = threshold - 0.15
            neg_threshold=neg_threshold,  # pyright: ignore[reportArgumentType]
            window_size_samples=512,
        )
        return speech_timestamps
