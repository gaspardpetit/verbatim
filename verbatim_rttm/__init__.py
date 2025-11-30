"""Lightweight RTTM reader/writer aligned to the NIST Rich Transcription specification."""

from .rttm import Annotation, Segment, load_rttm, loads_rttm, write_rttm
from .vttm import AudioRef, load_vttm, write_vttm

__all__ = ["Annotation", "Segment", "load_rttm", "loads_rttm", "write_rttm", "AudioRef", "load_vttm", "write_vttm"]
