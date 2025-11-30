from dataclasses import dataclass
from typing import Optional

Annotation = object  # pylint: disable=invalid-name


@dataclass
class SourceConfig:
    isolate: Optional[bool] = None
    diarize: Optional[int] = None
    diarization: Optional[Annotation] = None  # verbatim_rttm.Annotation expected at runtime
    diarization_file: Optional[str] = None  # legacy RTTM; when set, we derive from vttm
    diarization_strategy: str = "pyannote"
    vttm_file: Optional[str] = None
