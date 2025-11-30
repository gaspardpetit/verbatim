"""
Lightweight core package for transcription logic that can be shared independently
of the heavier verbatim package. Keeps dependencies minimal.
"""

from .interfaces import (
    GuessLanguageFn,
    LanguageDetectionRequest,
    LanguageDetectionResult,
    TranscriberProtocol,
    TranscriptionWindowResult,
    VadFn,
)
from .language import detect_language

__all__ = [
    "detect_language",
    "GuessLanguageFn",
    "LanguageDetectionRequest",
    "LanguageDetectionResult",
    "TranscriptionWindowResult",
    "TranscriberProtocol",
    "VadFn",
]
