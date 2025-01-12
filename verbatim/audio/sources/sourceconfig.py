from dataclasses import dataclass
from typing import Optional

from pyannote.core.annotation import Annotation


@dataclass
class SourceConfig:
    isolate: Optional[bool] = None
    diarize: Optional[int] = None
    diarization: Optional[Annotation] = None
    diarization_file: Optional[str] = None

    def __init__(
        self,
        isolate: Optional[bool] = None,
        diarize: Optional[int] = None,
        diarization_file: Optional[str] = None,
    ):
        self.isolate = isolate
        self.diarize = diarize
        self.diarization_file = diarization_file

        if self.diarize == "":
            self.diarize = 0
        elif self.diarize is not None:
            self.diarize = int(self.diarize)
