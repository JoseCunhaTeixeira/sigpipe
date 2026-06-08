from collections.abc import Sequence
from typing import Literal

from src.sigproc.algorithms.normalization.registry import NORMALIZATION_METHODS
from src.sigproc.base.stream import Stream
from src.sigproc.base.transformer import Transformer


class Normalize(Transformer):
    """
    Time normalization transformer.
    """

    def __init__(
        self,
        method: Literal["onebit"],
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        algorithm = NORMALIZATION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(NORMALIZATION_METHODS.keys())}"
            )

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = algorithm(
                stream=stream,
                **self.params,
            )
            streams_out.append(stream_out)

        return streams_out
