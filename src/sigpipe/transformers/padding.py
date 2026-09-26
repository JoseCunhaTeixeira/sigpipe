from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.padding.registry import PADDING_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Pad(Transformer[Stream, Stream]):
    """
    Padding transformer.
    """

    def __init__(
        self,
        method: Literal["none", "zeros"] = "zeros",
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = PADDING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown padding method '{self.method}'. "
                f"Available methods: {list(PADDING_METHODS.keys())}"
            )

        return [algorithm(stream=stream, **self.params) for stream in data]
