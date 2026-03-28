from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SYSTEMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "systems.yaml"
BENCHMARK_CONFIG_PATH = Path(__file__).resolve().parents[1] / "benchmark.yaml"
_ALLOWED_SYSTEM_KEYS = {"description", "mode", "fixed_primary_language", "overrides"}
_ALLOWED_SYSTEM_MODES = {"pipeline", "whisper_baseline", "whisper_mlx_baseline", "qwen_baseline"}


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_system_specs(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = (config_path or SYSTEMS_CONFIG_PATH).resolve()
    loaded = _read_yaml(path) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Systems config must be a mapping: {path}")

    systems: dict[str, dict[str, Any]] = {}
    for name, spec in loaded.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"System names must be non-empty strings: {path}")
        if not isinstance(spec, dict):
            raise ValueError(f"System '{name}' must map to an object in {path}")
        unknown = sorted(set(spec) - _ALLOWED_SYSTEM_KEYS)
        if unknown:
            raise ValueError(f"System '{name}' contains unknown keys {unknown} in {path}")
        description = spec.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"System '{name}' must define a non-empty description in {path}")
        mode = spec.get("mode")
        if mode in (None, ""):
            normalized_mode = "pipeline"
        elif isinstance(mode, str):
            normalized_mode = mode.strip()
        else:
            raise ValueError(f"System '{name}' mode must be a string in {path}")
        if normalized_mode not in _ALLOWED_SYSTEM_MODES:
            raise ValueError(f"System '{name}' has unsupported mode '{normalized_mode}' in {path}")
        overrides = spec.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"System '{name}' overrides must be a mapping in {path}")
        systems[name.strip()] = {
            "description": description.strip(),
            "mode": normalized_mode,
            "fixed_primary_language": bool(spec.get("fixed_primary_language", False)),
            "overrides": dict(overrides),
        }
    if not systems:
        raise ValueError(f"No systems defined in {path}")
    return systems


def load_benchmark_plan(
    plan_path: Path | None = None,
    *,
    systems_config_path: Path | None = None,
) -> dict[str, list[str]]:
    path = (plan_path or BENCHMARK_CONFIG_PATH).resolve()
    loaded = _read_yaml(path) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Benchmark config must be a mapping: {path}")
    languages = loaded.get("languages")
    systems = loaded.get("systems")
    if not isinstance(languages, list) or not all(isinstance(item, str) and item.strip() for item in languages):
        raise ValueError(f"Benchmark config languages must be a list of names: {path}")
    if not isinstance(systems, list) or not all(isinstance(item, str) and item.strip() for item in systems):
        raise ValueError(f"Benchmark config systems must be a list of names: {path}")

    known_systems = load_system_specs(systems_config_path)
    unknown = [name for name in systems if name not in known_systems]
    if unknown:
        raise ValueError(f"Benchmark config references unknown systems {unknown} in {path}")
    return {
        "languages": [item.strip() for item in languages],
        "systems": [item.strip() for item in systems],
    }
