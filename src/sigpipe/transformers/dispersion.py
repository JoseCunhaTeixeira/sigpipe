from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.dispersion.registry import DISPERSION_METHODS
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Dispersion(Transformer[Stream, Stream | DispersionImage]):
    """
    Dispersion transformer.
    """

    def __init__(
        self,
        method: Literal["none", "group", "phase"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream] | list[DispersionImage]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = DISPERSION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown dispersion method '{self.method}'. "
                f"Available methods: {list(DISPERSION_METHODS.keys())}"
            )

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
