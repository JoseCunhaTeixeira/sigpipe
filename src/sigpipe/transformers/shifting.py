from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.shifting.registry import SHIFTING_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Shift(Transformer[Stream, Stream]):
    """
    Time origin shifting transformer.
    """

    def __init__(
        self,
        method: Literal["none", "shift"] = "shift",
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = SHIFTING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown shifting method '{self.method}'. "
                f"Available methods: {list(SHIFTING_METHODS.keys())}"
            )

        return [algorithm(stream=stream, **self.params) for stream in data]
