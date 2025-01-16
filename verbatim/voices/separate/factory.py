# verbatim/voices/diarize/factory.py
from .separate import SeparationStrategy
from .pyannote import PyannoteSpeakerSeparation
from .channels import ChannelSeparation

def create_separator(strategy: str = "pyannote", **kwargs) -> SeparationStrategy:
    if strategy == "pyannote":
        return PyannoteSpeakerSeparation(**kwargs)

    if strategy == "channels":
        return ChannelSeparation(**kwargs)

    raise ValueError(f"Unknown separation strategy: {strategy}")
