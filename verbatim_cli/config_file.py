"""Config file loading and parser defaults application for CLIs."""

import argparse
import fnmatch
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from verbatim.transcript.format.writer import LanguageStyle, ProbabilityStyle, SpeakerStyle, TimestampStyle

OUTPUT_FLAGS = ("ass", "docx", "txt", "json", "md", "stdout", "stdout_nocolor")


def load_config_file(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config file format: {suffix}")


def _normalize_styles(cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(cfg)
    if "format_timestamp" in normalized and isinstance(normalized["format_timestamp"], str):
        normalized["format_timestamp"] = TimestampStyle[normalized["format_timestamp"]]
    if "format_speaker" in normalized and isinstance(normalized["format_speaker"], str):
        normalized["format_speaker"] = SpeakerStyle[normalized["format_speaker"]]
    if "format_probability" in normalized and isinstance(normalized["format_probability"], str):
        normalized["format_probability"] = ProbabilityStyle[normalized["format_probability"]]
    if "format_language" in normalized and isinstance(normalized["format_language"], str):
        normalized["format_language"] = LanguageStyle[normalized["format_language"]]
    return normalized


def _flatten_output(output_section: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if "formats" in output_section and isinstance(output_section["formats"], list):
        selected = set(output_section["formats"])
        for flag in OUTPUT_FLAGS:
            flat[flag] = flag.replace("_", "-") in selected or flag in selected
    else:
        for flag in OUTPUT_FLAGS:
            if flag in output_section:
                flat[flag] = output_section[flag]
    return flat


def _flatten_format(format_section: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key in ("timestamp", "speaker", "probability", "language"):
        fmt_key = f"format_{key}"
        if key in format_section:
            flat[fmt_key] = format_section[key]
    return flat


def flatten_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, val in profile.items():
        if key == "match":
            continue
        if key == "output" and isinstance(val, dict):
            flat.update(_flatten_output(val))
            continue
        if key == "format" and isinstance(val, dict):
            flat.update(_flatten_format(val))
            continue
        flat[key] = val
    return _normalize_styles(flat)


def _match_profile(profile: Dict[str, Any], filename: Optional[str]) -> bool:
    match = profile.get("match")
    if match is None:
        return True  # wildcard
    patterns = match.get("filename") if isinstance(match, dict) else None
    if not patterns:
        return True
    if filename is None:
        return False
    return any(fnmatch.fnmatch(Path(filename).name, pattern) for pattern in patterns)


def select_profile(config_data: Dict[str, Any], filename: Optional[str]) -> Dict[str, Any]:
    if not config_data:
        return {}
    profiles = config_data.get("profiles")
    if isinstance(profiles, list) and profiles:
        fallback: Optional[Dict[str, Any]] = None
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            if _match_profile(profile, filename):
                return flatten_profile(profile)
            if profile.get("match") is None and fallback is None:
                fallback = profile
        return flatten_profile(fallback) if fallback else {}
    # Treat entire config as single profile
    return flatten_profile(config_data)


def merge_args(base_defaults: Namespace, profile_overrides: Dict[str, Any], user_args: Namespace) -> Namespace:
    merged = vars(base_defaults).copy()
    merged.update(profile_overrides)
    merged.update(vars(user_args))
    return argparse.Namespace(**merged)
