from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from sigproc.base.pipeline import Pipeline

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Transformer[InputT, OutputT](ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def transform(self, data: Sequence[InputT]) -> Sequence[OutputT]:
        pass

    def __rshift__(self, other: Transformer | Pipeline) -> Pipeline:
        return Pipeline([self]) >> other
