from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.mutting.registry import MUTTING_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class Mute(Transformer[Stream, Stream]):
    """
    Mutting transformer.
    """

    def __init__(
        self,
        method: Literal["none", "mute"] = "mute",
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = MUTTING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown mutting method '{self.method}'. "
                f"Available methods: {list(MUTTING_METHODS.keys())}"
            )

        return [algorithm(stream=stream, **self.params) for stream in data]
