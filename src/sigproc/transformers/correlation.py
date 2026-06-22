from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.correlation.registry import (
    CORRELATION_METHODS,
)
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Correlate(Transformer[Stream, Stream]):
    """
    Correlation transformer.
    """

    def __init__(
        self,
        method: Literal["none", "cross"],
        virtual_source_index: int,
        **params: object,
    ) -> None:
        self.method = method
        self.virtual_source_index = virtual_source_index
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
            obj = algorithm(
                stream=stream,
                virtual_source_index=self.virtual_source_index,
                **self.params,
            )
            streams_out.extend(obj)

        return streams_out
