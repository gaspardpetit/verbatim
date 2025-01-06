from dataclasses import dataclass
from typing import Union

from pyannote.core.annotation import Annotation


@dataclass
class SourceConfig:
    isolate: Union[None, bool] = None
    diarize: Union[int, None] = None
    diarization: Annotation = None
    diarization_file: str = None

    def __init__(
        self,
        isolate: Union[None, bool] = None,
        diarize: Union[None, int] = None,
        diarization_file: Union[None, str] = None,
    ):
        self.isolate = isolate
        self.diarize = diarize
        self.diarization_file = diarization_file

        if self.diarize == "":
            self.diarize = 0
        elif self.diarize is not None:
            self.diarize = int(self.diarize)
