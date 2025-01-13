from dataclasses import dataclass
from typing import Optional

from pyannote.core.annotation import Annotation


@dataclass
class SourceConfig:
    isolate: Optional[bool] = None
    diarize: Optional[int] = None
    diarization: Optional[Annotation] = None
    diarization_file: Optional[str] = None
    diarization_strategy: str = "pyannote"
