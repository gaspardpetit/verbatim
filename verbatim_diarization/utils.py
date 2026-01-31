"""Utility helpers for diarization modules."""

from __future__ import annotations

import re
import unicodedata

# Allow word characters (unicode letters/digits), dots, underscores, and hyphens.
_URI_SAFE_PATTERN = re.compile(r"[^\w.\-]+", flags=re.UNICODE)


def sanitize_uri_component(value: str | None, *, fallback: str = "audio", max_length: int = 255) -> str:
    """Return a RTTM-safe token derived from ``value``.

    RTTM uses space-separated tokens, so whitespace or punctuation inside
    ``file_id`` causes downstream writers (such as :mod:`pyannote.core`) to
    raise ``ValueError``. This helper strips/normalizes text and replaces any
    disallowed characters with underscores so callers can safely write RTTMs
    even if the original filename contains spaces or Unicode punctuation.
    """

    normalized = unicodedata.normalize("NFKC", (value or "").strip())
    sanitized = _URI_SAFE_PATTERN.sub("_", normalized)
    sanitized = sanitized.strip("._-")

    if not sanitized:
        sanitized = fallback

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized
