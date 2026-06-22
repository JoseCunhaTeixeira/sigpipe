from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.beamforming.registry import BEAMFORMING_METHODS
from sigproc.base.beamforming import Beam
from sigproc.base.stream import Stream
from sigproc.base.transformer import Transformer


class Beamform(Transformer[Stream, Stream | Beam]):
    """
    Beamforming transformer.
    """

    def __init__(
        self,
        method: Literal["none", "cross"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(self, data: Sequence[Stream]) -> list[Stream] | list[Beam]:

        self.validate_sequence(data, Stream)

        if self.method == "none":
            return list(data)

        algorithm = BEAMFORMING_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown beamforming method '{self.method}'. "
                f"Available methods: {list(BEAMFORMING_METHODS.keys())}"
            )

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
