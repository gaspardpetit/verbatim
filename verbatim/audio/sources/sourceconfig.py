from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyannote.core.annotation import Annotation
else:  # pragma: no cover - type-only fallback to avoid runtime dependency
    Annotation = object  # pylint: disable=invalid-name


@dataclass
class SourceConfig:
    isolate: Optional[bool] = None
    diarize: Optional[int] = None
    diarization: Optional[Annotation] = None
    diarization_file: Optional[str] = None
    diarization_strategy: str = "pyannote"
