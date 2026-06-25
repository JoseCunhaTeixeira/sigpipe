from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.normalization.registry import NORMALIZATION_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Normalize(Transformer[Stream, Stream]):
    """
    Time normalization transformer.
    """

    def __init__(
        self,
        method: Literal["none", "onebit"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = NORMALIZATION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(NORMALIZATION_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = algorithm(
                stream=stream,
                **self.params,
            )
            streams_out.append(stream_out)

        return streams_out
