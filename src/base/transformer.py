from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Transformer(ABC, Generic[InputT, OutputT]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def transform(self, data: Sequence[InputT]) -> Sequence[OutputT]:
        pass
