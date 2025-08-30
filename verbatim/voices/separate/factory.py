# verbatim/voices/diarize/factory.py
from .channels import ChannelSeparation
from .pyannote import PyannoteSpeakerSeparation
from .separate import SeparationStrategy


def create_separator(strategy: str = "pyannote", **kwargs) -> SeparationStrategy:
    if strategy == "pyannote":
        return PyannoteSpeakerSeparation(**kwargs)

    if strategy == "channels":
        return ChannelSeparation(**kwargs)

    raise ValueError(f"Unknown separation strategy: {strategy}")
