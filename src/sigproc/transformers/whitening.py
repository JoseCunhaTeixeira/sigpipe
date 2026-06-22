from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.whitening.registry import WHITENING_METHODS
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Whiten(Transformer[Stream, Stream]):
    """
    Spectral whitening transformer.
    """

    def __init__(
        self,
        method: Literal[
            "none",
            "onebit",
            "onebit_apod",
            "savgol",
            "stft_onebit",
            "stft_savgol",
        ],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = WHITENING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown whitening method '{self.method}'. "
                f"Available methods: {list(WHITENING_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = algorithm(
                stream=stream,
                **self.params,
            )
            streams_out.append(stream_out)

        return streams_out
