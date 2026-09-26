from collections.abc import Sequence
from typing import Literal

from sigpipe.algorithms.residual_phase.registry import RESIDUAL_PHASE_METHODS
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class ArrivalResidualPhase(Transformer[Stream, Stream]):
    """
    Arrival residual phase transformer.
    """

    def __init__(
        self,
        method: Literal["none", "arrival"] = "arrival",
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = RESIDUAL_PHASE_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown residual phase method '{self.method}'. "
                f"Available methods: {list(RESIDUAL_PHASE_METHODS.keys())}"
            )

        return [algorithm(stream=stream, **self.params) for stream in data]
