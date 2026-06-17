from collections.abc import Sequence
from typing import Literal, TypeVar

from sigproc.algorithms.picking.registry import (
    DISPERSION_PICKING_METHODS,
    STREAM_PICKING_METHODS,
)
from sigproc.base.dispersion import DispersionImage
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer

T = TypeVar(
    "T",
    Stream,
    DispersionImage,
)


class Pick(Transformer):
    """
    Picking transformer.
    """

    def __init__(
        self,
        method: Literal["none", "maximum"],
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[T]) -> list[T]:

        if self.method == "none":
            return list(data)

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence, got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, (Stream, DispersionImage)) for s in data):
            raise TypeError("All elements must be Stream or DispersionImage")

        first = data[0]

        if isinstance(first, Stream):
            algorithm_stream = STREAM_PICKING_METHODS.get(self.method)
            if algorithm_stream is None:
                raise ValueError(
                    f"Unknown normalizing method '{self.method}'. "
                    f"Available methods: {list(STREAM_PICKING_METHODS.keys())}"
                )
            return [algorithm_stream(stream, **self.params) for stream in data]

        elif isinstance(first, DispersionImage):
            algorithm_stream = DISPERSION_PICKING_METHODS.get(self.method)
            if algorithm_stream is None:
                raise ValueError(
                    f"Unknown normalizing method '{self.method}'. "
                    f"Available methods: {list(DISPERSION_PICKING_METHODS.keys())}"
                )
            return [algorithm_stream(stream, **self.params) for stream in data]

        else:
            raise TypeError(f"No save handler for {type(first).__name__}")
