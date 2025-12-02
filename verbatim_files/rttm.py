import dataclasses
import os
from io import StringIO
from typing import IO, TYPE_CHECKING, Iterable, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from .vttm import AudioRef


@dataclasses.dataclass(order=True)
class Segment:
    """Simple diarization segment."""

    start: float
    end: float
    speaker: str
    file_id: str = ""
    channel: str = "1"
    orthography: str = "<NA>"
    subtype: str = "<NA>"
    confidence: Optional[float] = None
    slat: Optional[float] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Annotation:
    """Minimal Annotation for RTTM interoperability (NIST Rich Transcription)."""

    def __init__(self, segments: Optional[Iterable[Segment]] = None, file_id: str | None = None):
        self.file_id = file_id
        self._segments: List[Segment] = sorted(list(segments) if segments else [], key=lambda s: (s.start, s.end))

    def __len__(self) -> int:
        return len(self._segments)

    def itertracks(self, *, yield_label: bool = False) -> Iterator[Tuple[Segment, None, str]]:
        """Yield segments in start order, mimicking common diarization iterators."""
        for segment in sorted(self._segments, key=lambda s: (s.start, s.end)):
            yield (segment, None, segment.speaker) if yield_label else (segment, None, None)  # type: ignore

    def write_rttm(self, fp) -> None:
        """Write to an RTTM file-like object."""
        for line in _serialize_segments(self._segments):
            fp.write(line)

    def add(self, segment: Segment) -> None:
        self._segments.append(segment)
        self._segments.sort(key=lambda s: (s.start, s.end))

    @property
    def segments(self) -> List[Segment]:
        return list(self._segments)


def _parse_float(value: str) -> Optional[float]:
    if value in ("<NA>", "NA", ""):
        return None
    return float(value)


def _parse_segment(parts: List[str]) -> Segment:
    if len(parts) < 8:
        raise ValueError(f"Malformed RTTM line (expected >= 8 fields): {' '.join(parts)}")

    _, file_id, channel, start, duration, orthography, subtype, speaker, *rest = parts
    confidence = _parse_float(rest[0]) if len(rest) > 0 else None
    slat = _parse_float(rest[1]) if len(rest) > 1 else None

    return Segment(
        start=float(start),
        end=float(start) + float(duration),
        speaker=speaker if speaker != "<NA>" else "unknown",
        file_id=file_id,
        channel=channel,
        orthography=orthography,
        subtype=subtype,
        confidence=confidence,
        slat=slat,
    )


def load_rttm(path: str) -> Annotation:
    """Parse RTTM file into an Annotation (NIST RTTM format)."""
    with open(path, "r", encoding="utf-8") as fh:
        return _load_rttm_lines(fh)


def loads_rttm(text: str) -> Annotation:
    """Parse RTTM content provided as a string (NIST RTTM format)."""
    return _load_rttm_lines(StringIO(text))


def _load_rttm_lines(lines: Iterable[str]) -> Annotation:
    segments: List[Segment] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        parts = line.split()
        if parts[0].upper() != "SPEAKER":
            continue  # ignore non-speaker records for now
        segment = _parse_segment(parts)
        segments.append(segment)
    file_id = segments[0].file_id if segments else None
    return Annotation(segments=segments, file_id=file_id)


def _serialize_segments(segments: Iterable[Segment]) -> Iterator[str]:
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        conf = f"{seg.confidence:.3f}" if seg.confidence is not None else "<NA>"
        slat = f"{seg.slat:.3f}" if seg.slat is not None else "<NA>"
        yield (
            f"SPEAKER {seg.file_id} {seg.channel} {seg.start:.3f} {seg.duration:.3f} {seg.orthography} {seg.subtype} {seg.speaker} {conf} {slat}\n"
        )


def write_rttm(annotation: Annotation, dest: str | IO[str]) -> None:
    """Write an Annotation to an RTTM file or file-like object."""
    if isinstance(dest, str):
        with open(dest, "w", encoding="utf-8") as fh:
            for line in _serialize_segments(annotation.segments):
                fh.write(line)
    else:
        for line in _serialize_segments(annotation.segments):
            dest.write(line)


def _resolve_audio_refs(
    annotation: Annotation,
    audio_refs: Iterable["AudioRef"] | None,
    audio_path: str | None,
    channels: str | int,
) -> List["AudioRef"]:
    from .vttm import AudioRef  # pylint: disable=import-outside-toplevel

    provided = [AudioRef(id=str(ref.id), path=str(ref.path), channels=str(ref.channels)) for ref in (audio_refs or [])]
    if provided:
        return provided
    channel_spec = str(channels)
    if audio_path:
        audio_id = annotation.file_id or os.path.splitext(os.path.basename(audio_path))[0] or os.path.basename(audio_path)
        return [AudioRef(id=str(audio_id), path=audio_path, channels=channel_spec)]
    if annotation.file_id:
        return [AudioRef(id=str(annotation.file_id), path=str(annotation.file_id), channels=channel_spec)]
    raise ValueError("Cannot infer audio metadata for RTTM→VTTM conversion; provide audio_refs or audio_path.")


def rttm_to_vttm(
    rttm_path: str,
    vttm_path: str,
    *,
    audio_refs: Iterable["AudioRef"] | None = None,
    audio_path: str | None = None,
    channels: str | int = "1",
) -> Tuple[List["AudioRef"], Annotation]:
    from .vttm import write_vttm  # pylint: disable=import-outside-toplevel

    annotation = load_rttm(rttm_path)
    resolved = _resolve_audio_refs(annotation, audio_refs, audio_path, channels)
    write_vttm(vttm_path, audio=resolved, annotation=annotation)
    return resolved, annotation


def vttm_to_rttm(vttm_path: str, rttm_path: str) -> Annotation:
    from .vttm import load_vttm  # pylint: disable=import-outside-toplevel

    _audio_refs, annotation = load_vttm(vttm_path)
    write_rttm(annotation, rttm_path)
    return annotation
