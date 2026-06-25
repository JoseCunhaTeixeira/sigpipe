from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.detrending.registry import DETRENDING_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Detrend(Transformer[Stream, Stream]):
    """
    Detrending transformer.
    """

    def __init__(
        self,
        method: Literal["none", "linear", "constant"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = DETRENDING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown detrending method '{self.method}'. "
                f"Available methods: {list(DETRENDING_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = algorithm(
                stream=stream,
                **self.params,
            )
            streams_out.append(stream_out)

        return streams_out
