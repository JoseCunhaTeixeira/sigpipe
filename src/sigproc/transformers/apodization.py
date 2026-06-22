from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.apodization.registry import APODIZATION_METHODS
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Apodize(Transformer[Stream, Stream]):
    """
    Apodization transformer.
    """

    def __init__(
        self,
        method: Literal["none", "hanning"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = APODIZATION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown apodization method '{self.method}'. "
                f"Available methods: {list(APODIZATION_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = algorithm(
                stream=stream,
                **self.params,
            )
            streams_out.append(stream_out)

        return streams_out
