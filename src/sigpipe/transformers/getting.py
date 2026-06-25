from collections.abc import Sequence

from sigpipe.base.arrivals import TraceArrivals
from sigpipe.base.dispersion_curve import DispersionCurves
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.stream import Stream
from sigpipe.base.transformer import Transformer


class GetDispersionCurves(Transformer[DispersionImage, DispersionCurves]):
    """
    Get dispersion curves from Dispersion images transformer.
    """

    def transform(self, data: Sequence[DispersionImage]) -> list[DispersionCurves]:

        self.validate_sequence(data, DispersionImage)

        return [
            dispersion_image.dispersion_curves
            for dispersion_image in data
            if dispersion_image.dispersion_curves is not None
        ]


class GetArrivals(Transformer[Stream, tuple[TraceArrivals, ...]]):
    """
    Get arrivals from Streams transformer.
    """

    def transform(self, data: Sequence[Stream]) -> list[tuple[TraceArrivals, ...]]:

        self.validate_sequence(data, Stream)

        return [stream.arrivals for stream in data if stream.arrivals is not None]
