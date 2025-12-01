"""
VTTM: YAML-wrapped RTTM for self-contained diarization + audio references.
"""

import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .rttm import Annotation, Segment, loads_rttm, write_rttm


@dataclass
class AudioRef:
    """Reference to an audio asset used by the RTTM diarization."""

    id: str
    path: str
    channel: str | int = "1"


def load_vttm(path: str) -> Tuple[List[AudioRef], Annotation]:
    """Load a VTTM (YAML with embedded RTTM) file."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyYAML is required to load VTTM files. Install pyyaml.") from exc

    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

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
        "audio": [{"id": ref.id, "path": ref.path, "channel": ref.channel} for ref in audio],
        "rttm": LiteralScalarString("\n".join(rttm_lines) + ("\n" if rttm_lines else "")),
    }

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, width=sys.maxsize)


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
        channel = entry.get("channel", "1")
        audio_refs.append(AudioRef(id=str(audio_id), path=str(path), channel=str(channel)))
    return audio_refs


def _serialize_annotation(annotation: Annotation):
    """Yield RTTM lines from an Annotation."""
    # Reuse write_rttm logic via an in-memory buffer
    import io

    buffer = io.StringIO()
    write_rttm(annotation, buffer)  # type: ignore[arg-type]
    buffer.seek(0)
    for line in buffer:
        yield line
