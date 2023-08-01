from abc import ABC, abstractmethod


class Parser(ABC):
    @abstractmethod
    def parse(self, *args, **kwargs):
        pass


class Exporter(ABC):
    @abstractmethod
    def export(self, *args, **kwargs):
        pass
