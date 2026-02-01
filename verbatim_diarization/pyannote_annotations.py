from __future__ import annotations

from typing import TYPE_CHECKING, Union

from verbatim_files.rttm import Annotation, Segment

if TYPE_CHECKING:
    from pyannote.core.annotation import Annotation as PyannoteAnnotation
else:  # pragma: no cover - typing-only alias when pyannote is optional
    PyannoteAnnotation = object  # type: ignore[misc]


def to_rttm_annotation(annotation: Union[Annotation, "PyannoteAnnotation"]) -> Annotation:
    if isinstance(annotation, Annotation):
        return annotation
    if hasattr(annotation, "segments"):
        try:
            return Annotation(segments=list(annotation.segments), file_id=getattr(annotation, "file_id", None))
        except Exception as exc:  # pragma: no cover - defensive: unexpected segments shape
            raise TypeError("Unsupported diarization annotation segments") from exc
    if hasattr(annotation, "itertracks"):
        uri = getattr(annotation, "uri", None)
        file_id = str(uri) if uri else ""
        segments = [
            Segment(start=seg.start, end=seg.end, speaker=str(label), file_id=file_id)
            for seg, _track, label in annotation.itertracks(yield_label=True)  # type: ignore[attr-defined]
        ]
        return Annotation(segments=segments, file_id=file_id or None)
    raise TypeError("Unsupported diarization annotation type for RTTM serialization")
