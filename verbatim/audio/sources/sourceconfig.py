from dataclasses import dataclass
from typing import Optional

from pyannote.core.annotation import Annotation


@dataclass
class SourceConfig:
    isolate: Optional[bool] = None
    diarize: Optional[int] = None
    diarization: Optional[Annotation] = None
    diarization_file: Optional[str] = None
    diarization_strategy: str = None

    def __init__(
        self,
        isolate: Optional[bool] = None,
        diarize: Optional[int] = None,
        diarization_file: Optional[str] = None,
        diarization_strategy: str = "pyannote",
    ):
        self.isolate = isolate
        self.diarize = diarize
        self.diarization_file = diarization_file
        self.diarization_strategy = diarization_strategy

        if self.diarize == "":
            self.diarize = 0
        elif self.diarize is not None:
            self.diarize = int(self.diarize)
