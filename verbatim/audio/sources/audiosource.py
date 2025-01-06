from abc import abstractmethod

import numpy as np


class AudioStream:
    start_offset:int = 0
    def __init__(self, start_offset:int):
        self.start_offset = start_offset

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        self.close()
        return False

    @abstractmethod
    def has_more(self) -> bool:
        pass

    @abstractmethod
    def next_chunk(self, chunk_length=1) -> np.ndarray:
        pass

    @abstractmethod
    def close(self):
        pass

class AudioSource:
    def __init__(self):
        pass

    @abstractmethod
    def open(self) -> AudioStream:
        pass
