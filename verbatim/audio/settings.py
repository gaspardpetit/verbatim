from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioParams:
    """Parameters controlling audio frame sizing and rate.

    Attributes:
        sample_rate: Target sample rate for internal processing.
        frame_size: Number of samples per frame.
        max_window_frames: Maximum number of frames in an attention window.
    """

    sample_rate: int = 16000
    frame_size: int = 160
    max_window_frames: int = 3000

    @property
    def fps(self) -> int:
        """Frames per second derived from sample_rate and frame_size."""
        return self.sample_rate // self.frame_size


def get_audio_params() -> AudioParams:
    """Load audio parameters from environment variables if available."""
    sample_rate = int(os.getenv("VERBATIM_SAMPLE_RATE", "16000"))
    frame_size = int(os.getenv("VERBATIM_FRAME_SIZE", "160"))
    max_window_frames = int(os.getenv("VERBATIM_MAX_WINDOW_FRAMES", "3000"))
    return AudioParams(
        sample_rate=sample_rate,
        frame_size=frame_size,
        max_window_frames=max_window_frames,
    )


AUDIO_PARAMS = get_audio_params()
