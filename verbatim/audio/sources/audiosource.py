from abc import abstractmethod
from typing import Optional

from pyannote.core.annotation import Annotation
from numpy.typing import NDArray


class AudioStream:
    start_offset: int = 0
    diarization: Optional[Annotation] = None

    def __init__(self, start_offset: int, diarization: Optional[Annotation]):
        self.start_offset = start_offset
        self.diarization = diarization

    def __enter__(self) -> "AudioStream":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.close()
        return False

    @abstractmethod
    def has_more(self) -> bool:
        pass

    @abstractmethod
    def next_chunk(self, chunk_length=1) -> NDArray:
        pass

    @abstractmethod
    def close(self):
        pass


class AudioSource:
    source_name: str = ""

    def __init__(self, source_name: str):
        self.source_name = source_name

    @abstractmethod
    def open(self) -> AudioStream:
        pass
