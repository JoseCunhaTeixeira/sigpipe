from collections.abc import Sequence

from src.algorithms.beamforming.registry import BEAMFORMING_METHODS
from src.base.beamforming import Beam
from src.base.stream import Stream
from src.base.transformer import Transformer


class Beamform(Transformer):
    """
    Beamforming transformer.
    """

    def __init__(
        self,
        method: str,
        **params,
    ):
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Beam]:

        algorithm = BEAMFORMING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown normalizing method '{self.method}'. "
                f"Available methods: {list(BEAMFORMING_METHODS.keys())}"
            )

        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise TypeError(f"Expected Sequence[Stream], got {type(data).__name__}")

        if len(data) == 0:
            raise ValueError("Empty input sequence")

        if not all(isinstance(s, Stream) for s in data):
            raise TypeError("All elements must be Stream")

        beams_out: list[Beam] = []
        for stream in data:
            obj = algorithm(
                stream=stream,
                **self.params,
            )
            if isinstance(obj, Sequence):
                beams_out.extend(obj)
            else:
                beams_out.append(obj)

        return beams_out
