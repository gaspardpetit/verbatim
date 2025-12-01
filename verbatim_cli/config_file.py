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
DEFAULT_MATCH = ["*.wav", "*.mp3", "*.m4a", "*.mp4"]


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
    formats = None
    if "formats" in output_section and isinstance(output_section["formats"], list):
        formats = output_section["formats"]
    if "files" in output_section and isinstance(output_section["files"], list):
        formats = output_section["files"]
    if formats is not None:
        selected = set(formats)
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
            if isinstance(val, dict):
                filenames = val.get("filename")
                if filenames:
                    flat["match"] = filenames
            elif isinstance(val, list):
                flat["match"] = val
            continue
        if key == "include" and isinstance(val, dict):
            filenames = val.get("filename")
            if filenames:
                flat["match"] = filenames
            continue
        if key == "ignore" and isinstance(val, dict):
            filenames = val.get("filename")
            if filenames:
                flat["ignore"] = filenames
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
    includes = []
    excludes = []
    if isinstance(profile.get("include"), dict):
        includes = profile["include"].get("filename") or []
    if isinstance(profile.get("ignore"), dict):
        excludes = profile["ignore"].get("filename") or []

    if filename is None:
        return True
    name = Path(filename).name
    if includes:
        if not any(fnmatch.fnmatch(name, pat) for pat in includes):
            return False
    if excludes and any(fnmatch.fnmatch(name, pat) for pat in excludes):
        return False
    return True


def select_profile(config_data: Dict[str, Any], filename: Optional[str]) -> Dict[str, Any]:
    if not config_data:
        return {}

    base_defaults = flatten_profile({k: v for k, v in config_data.items() if k != "profiles"})
    if "match" not in base_defaults:
        base_defaults["match"] = DEFAULT_MATCH
    profiles = config_data.get("profiles")
    if filename is None and profiles:
        # For global defaults (no filename), ignore profile matching and return root defaults only.
        return base_defaults
    if isinstance(profiles, list) and profiles:
        fallback: Optional[Dict[str, Any]] = None
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            if _match_profile(profile, filename):
                merged = base_defaults.copy()
                merged.update(flatten_profile(profile))
                return merged
            if profile.get("match") is None and fallback is None:
                fallback = profile
        if fallback:
            merged = base_defaults.copy()
            merged.update(flatten_profile(fallback))
            return merged
        return base_defaults

    # Single-profile config; honor match at root
    if filename is not None and not _match_profile(config_data, filename):
        return {}
    merged = base_defaults.copy()
    merged.update(flatten_profile(config_data))
    return merged


def merge_args(base_defaults: Namespace, profile_overrides: Dict[str, Any], user_args: Namespace) -> Namespace:
    merged = vars(base_defaults).copy()
    merged.update(profile_overrides)
    for key, val in vars(user_args).items():
        # Only override when the user provided a value different from the parser default
        if key in merged and val == getattr(base_defaults, key, None):
            continue
        merged[key] = val
    # If match/ignore not specified by user or profile, ensure defaults
    if "match" not in merged or merged["match"] is None:
        merged["match"] = DEFAULT_MATCH
    if "ignore" not in merged or merged["ignore"] is None:
        merged["ignore"] = []
    return argparse.Namespace(**merged)
