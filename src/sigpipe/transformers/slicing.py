from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.segmentation.registry import SEGMENTATION_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Slice(Transformer[Stream, Stream]):
    """
    Slicing transformer.
    """

    def __init__(
        self,
        method: Literal["none", "slice"] = "slice",
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = SEGMENTATION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown slicing method '{self.method}'. "
                f"Available methods: {list(SEGMENTATION_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            streams_out.extend(algorithm(stream=stream, **self.params))
        return streams_out
