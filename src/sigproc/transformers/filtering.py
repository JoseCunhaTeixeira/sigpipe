from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.filtering.registry import FILTERING_METHODS
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Filter(Transformer[Stream, Stream]):
    """
    Filtering transformer.
    """

    def __init__(
        self,
        method: Literal["none", "iir"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = FILTERING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown filtering method '{self.method}'. "
                f"Available methods: {list(FILTERING_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = algorithm(
                stream=stream,
                **self.params,
            )
            streams_out.append(stream_out)

        return streams_out
