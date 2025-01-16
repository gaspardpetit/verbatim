# verbatim/voices/diarize/factory.py
from .separate import SeparationStrategy
from .pyannote import PyannoteSpeakerSeparation


def create_separator(strategy: str = "pyannote", **kwargs) -> SeparationStrategy:
    if strategy == "pyannote":
        return PyannoteSpeakerSeparation(**kwargs)

    raise ValueError(f"Unknown separation strategy: {strategy}")
