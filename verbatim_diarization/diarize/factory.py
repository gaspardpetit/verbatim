"""Factory for diarization strategies with optional interactive HF-token prompt.

If no Hugging Face token is provided and we're not in offline mode, this module
will prompt for a token when running in an interactive terminal. For
non-interactive environments, behavior remains unchanged (raises a clear error).
"""

import os
import sys
from getpass import getpass
from typing import Optional

# pylint: disable=import-outside-toplevel,broad-exception-caught
from verbatim_diarization.diarize.base import DiarizationStrategy


def create_diarizer(strategy: str = "pyannote", device: str = "cpu", huggingface_token: Optional[str] = None, **kwargs) -> DiarizationStrategy:
    """
    Factory function to create diarization strategy instances.

    Args:
        strategy: Name of the strategy to use ('pyannote', 'energy', 'channel')
        device: Device to use for PyAnnote ('cpu' or 'cuda')
        huggingface_token: Token for PyAnnote Hub
        **kwargs: Additional strategy-specific parameters

    Returns:
        DiarizationStrategy instance
    """
    if strategy in ("pyannote", "separate"):
        offline_env = os.getenv("VERBATIM_OFFLINE", "0").lower() in ("1", "true", "yes")
        token = (huggingface_token or os.getenv("HUGGINGFACE_TOKEN") or "").strip()

        if not token and not offline_env and sys.stdin.isatty() and sys.stdout.isatty():
            print("Pyannote models require a Hugging Face access token.")
            print("Create one at https://huggingface.co/settings/tokens and ensure gated model access.")
            try:
                entered = getpass("Enter HUGGINGFACE_TOKEN (starts with hf_): ")
            except Exception:
                entered = ""
            if entered:
                token = entered.strip()
                os.environ["HUGGINGFACE_TOKEN"] = token
                print("Token received. Tip: export HUGGINGFACE_TOKEN to avoid prompts next time.")

        if not token and not offline_env:
            raise ValueError("huggingface_token is required for PyAnnote diarization. Set HUGGINGFACE_TOKEN or run interactively to be prompted.")

        # When offline, allow missing token (loading from local cache only)
        if strategy == "pyannote":
            from verbatim_diarization.pyannote import PyAnnoteDiarization

            return PyAnnoteDiarization(device=device, huggingface_token=token)

        from verbatim_diarization.pyannote import PyAnnoteSeparationDiarization

        return PyAnnoteSeparationDiarization(device=device, huggingface_token=token)

    if strategy == "energy":
        from verbatim_diarization.stereo import EnergyDiarization

        return EnergyDiarization(**kwargs)

    if strategy == "channel":
        from verbatim_diarization.channel import ChannelDiarization

        return ChannelDiarization(**kwargs)

    raise ValueError(f"Unknown diarization strategy: {strategy}")
