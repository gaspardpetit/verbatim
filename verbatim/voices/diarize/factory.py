# verbatim/voices/diarize/factory.py
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
        if huggingface_token is None:
            raise ValueError("huggingface_token is required for PyAnnote diarization")
        return PyAnnoteDiarization(device=device, huggingface_token=huggingface_token)
    elif strategy == "stereo":
        return StereoDiarization(**kwargs)
    else:
        raise ValueError(f"Unknown diarization strategy: {strategy}")
