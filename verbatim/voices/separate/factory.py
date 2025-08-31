"""Factory for separation strategies with optional interactive HF-token prompt."""
import os
import sys
from getpass import getpass

from .channels import ChannelSeparation
from .pyannote import PyannoteSpeakerSeparation
from .separate import SeparationStrategy


def create_separator(strategy: str = "pyannote", **kwargs) -> SeparationStrategy:
    if strategy == "pyannote":
        offline_env = os.getenv("VERBATIM_OFFLINE", "0").lower() in ("1", "true", "yes")
        token = (kwargs.get("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN") or "").strip()  # type: ignore[call-overload]
        if not token and not offline_env and sys.stdin.isatty() and sys.stdout.isatty():
            print("Pyannote separation requires a Hugging Face access token.")
            print("Create one at https://huggingface.co/settings/tokens and ensure gated model access.")
            try:
                entered = getpass("Enter HUGGINGFACE_TOKEN (starts with hf_): ")
            except Exception:
                entered = ""
            if entered:
                token = entered.strip()
                os.environ["HUGGINGFACE_TOKEN"] = token
                print("Token received. Tip: export HUGGINGFACE_TOKEN to avoid prompts next time.")
        if token:
            kwargs["huggingface_token"] = token
        return PyannoteSpeakerSeparation(**kwargs)

    if strategy == "channels":
        return ChannelSeparation(**kwargs)

    raise ValueError(f"Unknown separation strategy: {strategy}")
