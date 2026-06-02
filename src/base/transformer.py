from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar, Union

from src.base.pipeline import Pipeline

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Transformer(ABC, Generic[InputT, OutputT]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def transform(self, data: Sequence[InputT]) -> Sequence[OutputT]:
        pass

    def __rshift__(self, other: Union["Transformer", "Pipeline"]):
        return Pipeline([self]) >> other
