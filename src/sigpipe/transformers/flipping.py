from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.flipping.registry import FLIPPING_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Flip(Transformer[Stream, Stream]):
    """
    Flip transformer.
    """

    def __init__(
        self,
        method: Literal["none", "flip"] = "flip",
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = FLIPPING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown flipping method '{self.method}'. "
                f"Available methods: {list(FLIPPING_METHODS.keys())}"
            )

        return [algorithm(stream=stream, **self.params) for stream in data]
