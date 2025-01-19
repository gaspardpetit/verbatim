from abc import ABC, abstractmethod


class IdProvider(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def next(self) -> str:
        pass


class CounterIdProvider(IdProvider):
    prefix: str
    suffix: str
    counter: int

    def __init__(self, prefix: str = "", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix
        self.counter = 0

    def next(self) -> str:
        self.counter += 1
        return f"{self.prefix}{self.counter}{self.suffix}"
