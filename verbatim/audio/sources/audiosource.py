from abc import abstractmethod

import numpy as np


class AudioSource:
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
