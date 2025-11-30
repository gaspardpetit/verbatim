"""
Core scaffolding for shared types and helpers.

These stay light-weight so they can be lifted into a dedicated verbatim-core
package later without pulling heavy dependencies.
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
