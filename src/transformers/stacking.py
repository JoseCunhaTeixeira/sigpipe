from collections.abc import Sequence
from typing import TypeVar

from src.algorithms.stacking.registry import STACKING_HANDLERS
from src.base.transformer import Transformer

T = TypeVar("T")


class Stack(Transformer):
    """
    Stacking transformer.
    """

    def __init__(
        self,
        method: str,
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[T]) -> Sequence[T]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(x, type(data[0])) for x in data):
            raise TypeError("All elements must have the same type")

        first = data[0]
        handler = STACKING_HANDLERS.get(type(first))
        if handler is None:
            raise TypeError(f"No save handler for {type(first).__name__}")

        algorithm = handler.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown stacking method '{self.method}'. "
                f"Available methods: {list(handler.keys())}"
            )

        stream_out = algorithm(
            data,
            **self.params,
        )

        return (stream_out,)
