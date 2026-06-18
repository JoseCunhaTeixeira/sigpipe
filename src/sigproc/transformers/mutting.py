from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.mutting.mutting import mute
from sigproc.algorithms.mutting.registry import MUTTING_METHODS
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Mute(Transformer):
    """
    Mutting transformer.
    """

    def __init__(
        self,
        method: Literal["none", "mute"] = "mute",
        tmin: float | None = None,
        tmax: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        taper: int = 0,
    ) -> None:
        self.method = method
        self.tmin = tmin
        self.tmax = tmax
        self.vmin = vmin
        self.vmax = vmax
        self.taper = taper

    def transform(self, data: Sequence[Stream]) -> list[Stream]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        if self.method == "none":
            return list(data)

        algorithm = MUTTING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(MUTTING_METHODS.keys())}"
            )

        streams_out: list[Stream] = []
        for stream in data:
            stream_out = mute(
                stream=stream,
                tmin=self.tmin,
                tmax=self.tmax,
                vmin=self.vmin,
                vmax=self.vmax,
                taper=self.taper,
            )
            streams_out.append(stream_out)

        return streams_out
