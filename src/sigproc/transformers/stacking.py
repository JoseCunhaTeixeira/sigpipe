from collections.abc import Sequence
from typing import Literal, cast

from sigproc.algorithms.stacking.registry import (
    DISPERSION_IMAGE_STACKING_METHODS,
    STREAM_STACKING_METHODS,
)
from sigproc.base.dispersion_image import DispersionImage
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Stack[T: (Stream, DispersionImage)](Transformer[T, T]):
    """
    Stacking transformer.
    """

    def __init__(
        self,
        method: Literal["none", "linear", "root", "phase_weighted"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(
        self,
        data: Sequence[T],
    ) -> list[T]:

        self.validate_homogeneous_sequence(data)

        if self.method == "none":
            return list(data)

        first = data[0]

        if isinstance(first, Stream):
            algorithm_stream = STREAM_STACKING_METHODS.get(self.method)
            if algorithm_stream is None:
                raise ValueError(
                    f"Unknown stream stacking method '{self.method}'. "
                    f"Available methods: {list(STREAM_STACKING_METHODS.keys())}"
                )
            return cast(list[T], [algorithm_stream(data, **self.params)])

        if isinstance(first, DispersionImage):
            algorithm_image = DISPERSION_IMAGE_STACKING_METHODS.get(self.method)
            if algorithm_image is None:
                raise ValueError(
                    f"Unknown dispersion image stacking method '{self.method}'. "
                    f"Available methods: {list(DISPERSION_IMAGE_STACKING_METHODS.keys())}"
                )
            return cast(list[T], [algorithm_image(data, **self.params)])

        raise TypeError(f"No stacking handler for {type(first).__name__}")
