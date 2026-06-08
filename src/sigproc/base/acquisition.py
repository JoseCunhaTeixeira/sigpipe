from dataclasses import dataclass, field

import numpy as np

from src.sigproc.base.coordinate import Coordinate


@dataclass(slots=True, frozen=True)
class Acquisition:
    source: Coordinate
    receivers: tuple[Coordinate, ...]

    offsets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        offsets = compute_offsets(self.source, self.receivers)
        object.__setattr__(self, "offsets", offsets)
        offsets.setflags(write=False)

    def __eq__(self, other):
        if not isinstance(other, Acquisition):
            return NotImplemented

        return (
            self.source == other.source
            and self.receivers == other.receivers
            and np.array_equal(
                self.offsets,
                other.offsets,
            )
        )

    @property
    def is_unknown(self) -> bool:
        values = [self.source.x, self.source.y, self.source.z]
        for receiver in self.receivers:
            values.extend([receiver.x, receiver.y, receiver.z])
        return any(np.isnan(v) for v in values)


def compute_offsets(
    source: Coordinate, receivers: tuple[Coordinate, ...]
) -> np.ndarray:
    return np.array(
        [source.distance_to(r) for r in receivers],
        dtype=np.float32,
    )


UNKNOWN_ACQUISITION = Acquisition(
    source=Coordinate(np.nan, np.nan, np.nan),
    receivers=(Coordinate(np.nan, np.nan, np.nan),),
)
