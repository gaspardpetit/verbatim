# verbatim/voices/diarize/factory.py
import os
from typing import Optional

from .base import DiarizationStrategy
from .pyannote import PyAnnoteDiarization
from .stereo import StereoDiarization


def create_diarizer(strategy: str = "pyannote", device: str = "cpu", huggingface_token: Optional[str] = None, **kwargs) -> DiarizationStrategy:
    """
    Factory function to create diarization strategy instances.

    Args:
        strategy: Name of the strategy to use ('pyannote' or 'stereo')
        device: Device to use for PyAnnote ('cpu' or 'cuda')
        huggingface_token: Token for PyAnnote Hub
        **kwargs: Additional strategy-specific parameters

    Returns:
        DiarizationStrategy instance
    """
    if strategy == "pyannote":
        offline_env = os.getenv("VERBATIM_OFFLINE", "0").lower() in ("1", "true", "yes")
        if huggingface_token is None and not offline_env:
            raise ValueError("huggingface_token is required for PyAnnote diarization when not offline")
        # When offline, allow missing token (loading from local cache only)
        return PyAnnoteDiarization(device=device, huggingface_token=huggingface_token or "")
    elif strategy == "stereo":
        return StereoDiarization(**kwargs)
    else:
        raise ValueError(f"Unknown diarization strategy: {strategy}")
