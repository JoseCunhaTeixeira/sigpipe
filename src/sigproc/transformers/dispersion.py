from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.dispersion.registry import DISPERSION_METHODS
from sigproc.base.dispersion import DispersionImage
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Dispersion(Transformer):
    """
    Dispersion transformer.
    """

    def __init__(
        self,
        method: Literal["none", "group", "phase"],
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream] | list[DispersionImage]:

        algorithm = DISPERSION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(DISPERSION_METHODS.keys())}"
            )

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        if self.method == "none":
            return list(data)

        dispersion_images_out: list[DispersionImage] = []
        for stream in data:
            obj = algorithm(
                stream=stream,
                **self.params,
            )
            if isinstance(obj, Sequence):
                dispersion_images_out.extend(obj)
            else:
                dispersion_images_out.append(obj)

        return dispersion_images_out
