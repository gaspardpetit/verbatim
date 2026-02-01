"""
VTTM: YAML-wrapped RTTM for self-contained diarization + audio references.
"""
# pylint: disable=import-outside-toplevel,broad-exception-caught,no-name-in-module

import io
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

from verbatim.cache import get_default_cache

from .rttm import Annotation, loads_rttm, write_rttm

ChannelSpec = Union[str, int, None]


def normalize_channel_spec(channels: ChannelSpec) -> ChannelSpec:
    """Validate and normalize a channel specification.

    Returns the sanitized value (strip whitespace for strings) or raises ValueError
    if the spec is malformed. Use ``None`` to indicate that all channels should be used.
    """

    if channels is None:
        return None
    if isinstance(channels, int):
        if channels < 0:
            raise ValueError("Channel index must be >= 0")
        return channels
    if not isinstance(channels, str):
        raise TypeError("Channel specification must be str, int, or None")

    spec = channels.strip()
    if not spec:
        raise ValueError("Channel specification cannot be empty; use None for all channels")

    for part in spec.split(","):
        token = part.strip()
        if not token:
            raise ValueError("Malformed channel specification: consecutive commas detected")
        if "-" in token:
            left, right = token.split("-", 1)
            if not left or not right:
                raise ValueError(f"Malformed channel range '{token}'")
            if not (left.isdigit() and right.isdigit()):
                raise ValueError(f"Channel range '{token}' must contain integers")
            start = int(left)
            end = int(right)
            if start < 0 or end < 0:
                raise ValueError("Channel indices must be >= 0")
            if end < start:
                raise ValueError(f"Channel range '{token}' must be ascending")
        else:
            if not token.isdigit():
                raise ValueError(f"Channel index '{token}' must be a non-negative integer")
            if int(token) < 0:
                raise ValueError("Channel indices must be >= 0")

    return spec


@dataclass
class AudioRef:
    """Reference to an audio asset used by the RTTM diarization."""

    id: str
    path: str
    channels: ChannelSpec = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", normalize_channel_spec(self.channels))


def load_vttm(path: str) -> Tuple[List[AudioRef], Annotation]:
    """Load a VTTM (YAML with embedded RTTM) file."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyYAML is required to load VTTM files. Install pyyaml.") from exc

    cache = get_default_cache()
    cached = cache.get_text(path) if cache else None
    if cached is None:
        with open(path, "r", encoding="utf-8") as fh:
            cached = fh.read()
        if cache:
            cache.set_text(path, cached)
    doc = yaml.safe_load(cached) or {}

    audio_entries = _parse_audio(doc.get("audio"))
    rttm_raw = doc.get("rttm", "")
    rttm_text = "\n".join(rttm_raw) if isinstance(rttm_raw, list) else str(rttm_raw)
    annotation = loads_rttm(rttm_text)
    return audio_entries, annotation


def write_vttm(path: str, *, audio: Iterable[AudioRef], annotation: Annotation) -> None:
    """Write a VTTM file with audio references and embedded RTTM."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyYAML is required to write VTTM files. Install pyyaml.") from exc

    rttm_lines = []
    for line in _serialize_annotation(annotation):
        rttm_lines.append(line.rstrip("\n"))

    try:
        from yaml.scalarstring import LiteralScalarString  # type: ignore
    except Exception:  # pragma: no cover - fallback if PyYAML changes
        LiteralScalarString = str  # type: ignore

    payload = {
        "audio": [{"id": ref.id, "path": ref.path, "channels": ref.channels} for ref in audio],
        "rttm": LiteralScalarString("\n".join(rttm_lines) + ("\n" if rttm_lines else "")),
    }

    rendered = yaml.safe_dump(payload, sort_keys=False, width=sys.maxsize)
    cache = get_default_cache()
    if cache:
        cache.set_text(path, rendered)
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(rendered)


def _parse_audio(raw_audio) -> List[AudioRef]:
    if raw_audio is None:
        return []
    if isinstance(raw_audio, dict):
        raw_audio = [raw_audio]
    audio_refs: List[AudioRef] = []
    for entry in raw_audio:
        if not isinstance(entry, dict):
            continue
        audio_id = entry.get("id") or entry.get("file") or entry.get("name")
        path = entry.get("path") or entry.get("file_path")
        if not audio_id or not path:
            continue
        channels = entry.get("channels", entry.get("channel"))
        audio_refs.append(AudioRef(id=str(audio_id), path=str(path), channels=channels))
    return audio_refs


def _serialize_annotation(annotation: Annotation):
    """Yield RTTM lines from an Annotation."""
    # Reuse write_rttm logic via an in-memory buffer
    buffer = io.StringIO()
    write_rttm(annotation, buffer)  # type: ignore[arg-type]
    buffer.seek(0)
    yield from buffer
