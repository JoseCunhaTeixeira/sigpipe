from collections.abc import Sequence
from typing import Literal, TypeVar

from sigproc.algorithms.stacking.registry import (
    DISPERSION_IMAGE_STACKING_METHODS,
    STREAM_STACKING_METHODS,
)
from sigproc.base.dispersion import DispersionImage
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer

T = TypeVar(
    "T",
    Stream,
    DispersionImage,
)


class Stack(Transformer):
    """
    Stacking transformer.
    """

    def __init__(
        self,
        method: Literal["none", "linear", "root", "phase_weighted"],
        **params,
    ):
        self.method = method
        self.params = params

    def transform(
        self,
        data: Sequence[T],
    ) -> list[T]:

        if self.method == "none":
            return list(data)

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(x, type(data[0])) for x in data):
            raise TypeError("All elements must have the same type")

        first = data[0]

        if isinstance(first, Stream):
            algorithm_stream = STREAM_STACKING_METHODS.get(self.method)
            if algorithm_stream is None:
                raise ValueError(
                    f"Unknown normalizing method '{self.method}'. "
                    f"Available methods: {list(STREAM_STACKING_METHODS.keys())}"
                )
            return [algorithm_stream(data, **self.params)]

        elif isinstance(first, DispersionImage):
            algorithm_image = DISPERSION_IMAGE_STACKING_METHODS.get(self.method)
            if algorithm_image is None:
                raise ValueError(
                    f"Unknown normalizing method '{self.method}'. "
                    f"Available methods: {list(DISPERSION_IMAGE_STACKING_METHODS.keys())}"
                )
            return [algorithm_image(data, **self.params)]

        else:
            raise TypeError(f"No save handler for {type(first).__name__}")
