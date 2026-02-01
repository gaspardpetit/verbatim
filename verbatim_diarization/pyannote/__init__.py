# pylint: disable=wrong-import-position
import os

# Disable pyannote telemetry unless caller opts in explicitly via env.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

from .diarize import PyAnnoteDiarization, PyAnnoteSeparationDiarization  # noqa: E402
from .separate import PyannoteSpeakerSeparation  # noqa: E402

__all__ = ["PyAnnoteDiarization", "PyAnnoteSeparationDiarization", "PyannoteSpeakerSeparation"]
