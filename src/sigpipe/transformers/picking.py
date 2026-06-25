from collections.abc import Sequence
from typing import Literal, cast

from sigpipe.algorithms.picking.registry import (
    DISPERSION_PICKING_METHODS,
    STREAM_PICKING_METHODS,
)
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Pick[T: (Stream, DispersionImage)](Transformer[T, T]):
    """
    Picking transformer.
    """

    def __init__(
        self,
        method: Literal["none", "maximum"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[T]) -> list[T]:

        self.validate_sequence(data, Stream, DispersionImage)

        if self.method == "none":
            return list(data)

        first = data[0]

        if isinstance(first, Stream):
            algorithm_stream = STREAM_PICKING_METHODS.get(self.method)
            if algorithm_stream is None:
                raise ValueError(
                    f"Unknown stream picking method '{self.method}'. "
                    f"Available methods: {list(STREAM_PICKING_METHODS.keys())}"
                )
            return cast(list[T], [algorithm_stream(stream, **self.params) for stream in data])

        if isinstance(first, DispersionImage):
            algorithm_stream = DISPERSION_PICKING_METHODS.get(self.method)
            if algorithm_stream is None:
                raise ValueError(
                    f"Unknown dispersion picking method '{self.method}'. "
                    f"Available methods: {list(DISPERSION_PICKING_METHODS.keys())}"
                )
            return cast(list[T], [algorithm_stream(stream, **self.params) for stream in data])

        raise TypeError(f"No picking handler for {type(first).__name__}")
