from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.correlation.registry import (
    CORRELATION_METHODS,
)
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Correlate(Transformer[Stream, Stream]):
    """
    Correlation transformer.
    """

    def __init__(
        self,
        method: Literal["none", "cross"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = CORRELATION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown correlation method '{self.method}'. "
                f"Available methods: {list(CORRELATION_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            obj = algorithm(stream=stream, **self.params)
            streams_out.extend(obj)

        return streams_out
