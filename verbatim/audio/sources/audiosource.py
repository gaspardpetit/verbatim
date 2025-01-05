from abc import abstractmethod

import numpy as np

class AudioSource:
    start_offset:int # start time in samples
    input_source:str # filename or representative name

    def __init__(self, name:str, start_offset:int = 0):
        self.input_source = name
        self.start_offset = start_offset

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
