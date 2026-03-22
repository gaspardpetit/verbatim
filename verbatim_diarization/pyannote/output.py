"""Helpers for normalizing pyannote pipeline outputs."""

from typing import Any


def select_speaker_diarization(output: Any) -> Any:
    """Prefer exclusive diarization when pyannote exposes it."""
    if hasattr(output, "exclusive_speaker_diarization"):
        exclusive = output.exclusive_speaker_diarization  # type: ignore[attr-defined]
        if exclusive is not None:
            return exclusive

    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization  # type: ignore[attr-defined]
        if diarization is not None:
            return diarization

    return output
