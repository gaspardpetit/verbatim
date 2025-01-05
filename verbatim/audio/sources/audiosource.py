from abc import abstractmethod

import numpy as np

class AudioSource:
    start_offset:int = 0 # start time in samples
    def __init__(self):
        pass

    @abstractmethod
    def next_chunk(self, chunk_length=1) -> np.ndarray:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def has_more(self) -> bool:
        pass
