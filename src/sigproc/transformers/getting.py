from collections.abc import Sequence

from sigproc.base.arrivals import TraceArrivals
from sigproc.base.dispersion import DispersionCurves, DispersionImage
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class GetDispersionCurves(Transformer):
    """
    Get dispersion curves from Dispersion images transformer.
    """

    def transform(self, data: Sequence[DispersionImage]) -> list[DispersionCurves]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence, got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, DispersionImage) for s in data):
            raise TypeError("All elements must be DispersionImage")

        return [
            dispersion_image.dispersion_curves
            for dispersion_image in data
            if dispersion_image.dispersion_curves is not None
        ]


class GetArrivals(Transformer):
    """
    Get arrivals from Streams transformer.
    """

    def transform(self, data: Sequence[Stream]) -> list[tuple[TraceArrivals, ...]]:

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence, got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        return [stream.arrivals for stream in data if stream.arrivals is not None]
