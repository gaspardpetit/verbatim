from abc import ABC, abstractmethod

# pylint: disable=unused-argument
class Filter(ABC):
    @abstractmethod
    def execute(self, **kwargs: dict):
        ...
    def load(self, **kwargs: dict):
        ...

    def unload(self, **kwargs: dict):
        ...
        