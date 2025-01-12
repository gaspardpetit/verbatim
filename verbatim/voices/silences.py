import logging
from abc import abstractmethod
from typing import Dict, List
import warnings

from numpy.typing import NDArray
import torch

from silero_vad import get_speech_timestamps, load_silero_vad

# Configure logger
LOG = logging.getLogger(__name__)


class VoiceActivityDetection:
    @abstractmethod
    def find_activity(
        self,
        audio: NDArray,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[Dict[str, int]]:
        pass


class SileroVoiceActivityDetection(VoiceActivityDetection):
    def __init__(self):
        with warnings.catch_warnings():
            # suppresses silero_vad\model.py:15: DeprecationWarning: path is deprecated. Use files() instead.
            # Refer to https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy for migration advice.
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.model = load_silero_vad()

    def find_activity(
        self,
        audio: NDArray,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[Dict[str, int]]:
        audio_tensor = torch.from_numpy(audio).float()
        speech_timestamps = get_speech_timestamps(
            audio=audio_tensor,
            model=self.model,
            threshold=0.25,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=250,
            return_seconds=False,
            visualize_probs=False,
            progress_tracking_callback=None,  # pyright: ignore[reportArgumentType]
            neg_threshold=None,  # pyright: ignore[reportArgumentType]
            window_size_samples=512,
        )
        return speech_timestamps
