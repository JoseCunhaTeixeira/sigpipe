from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from sigpipe.base.pipeline import Pipeline


class Transformer[InputT, OutputT](ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def transform(self, data: Sequence[InputT]) -> Sequence[OutputT]:
        pass

    def __rshift__(self, other: Transformer[Any, Any] | Pipeline) -> Pipeline:
        return Pipeline([self]) >> other

    @staticmethod
    def validate_sequence(data: object, *expected_types: type) -> None:
        """Raise unless `data` is a non-empty Sequence containing only the given types."""
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence, got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if expected_types and not all(isinstance(item, expected_types) for item in data):
            names = " or ".join(t.__name__ for t in expected_types)
            raise TypeError(f"All elements must be {names}")

    @staticmethod
    def validate_homogeneous_sequence(data: object) -> None:
        """Raise unless `data` is a non-empty Sequence whose elements all share the same type."""
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence, got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(x, type(data[0])) for x in data):
            raise TypeError("All elements must have the same type")
