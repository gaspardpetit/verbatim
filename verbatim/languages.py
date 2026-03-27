"""Shared language normalization helpers.

These helpers provide the repository-wide policy for mapping raw language labels
into the canonical codes used by Verbatim and the SwitchLingua benchmark.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from langcodes import Language

# Human-readable names and dataset labels that should collapse to the canonical
# codes used across the repo.
LANGUAGE_NAME_ALIASES: dict[str, str] = {
    "arabic": "ar",
    "bengali": "bn",
    "cantonese": "yue",
    "chinese": "zh",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "filipino": "fil",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "macedonian": "mk",
    "malay": "ms",
    "mandarin": "zh",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "spanish": "es",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "vietnamese": "vi",
}

# Repo-level canonicalization policy. langcodes deliberately preserves some
# distinctions such as cmn vs zh; for benchmark/runtime allowlist matching we
# want the broader zh code.
LANGUAGE_CODE_ALIASES: dict[str, str] = {
    "cmn": "zh",
}


def normalize_language(value: Any, *, aliases: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Normalize a language label or code to the repo's canonical code.

    Returns ``None`` for empty or unknown values.
    """

    if value in (None, ""):
        return None

    candidate = str(value).strip()
    if not candidate:
        return None

    alias_map = dict(LANGUAGE_CODE_ALIASES)
    if aliases:
        alias_map.update(aliases)

    lowered = candidate.lower()
    mapped = alias_map.get(lowered) or LANGUAGE_NAME_ALIASES.get(lowered)
    if mapped:
        return mapped

    try:
        normalized = Language.get(candidate).language
    except Exception:
        return None

    if not normalized or normalized == "und":
        return None

    normalized = normalized.lower()
    return alias_map.get(normalized, normalized)
