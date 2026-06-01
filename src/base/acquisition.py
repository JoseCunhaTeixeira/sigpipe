from dataclasses import dataclass, field

import numpy as np

from base.coordinate import Coordinate


@dataclass(slots=True, frozen=True)
class Acquisition:
    source: Coordinate
    receivers: tuple[Coordinate, ...]

    offsets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        offsets = compute_offsets(self.source, self.receivers)
        object.__setattr__(self, "offsets", offsets)
        offsets.setflags(write=False)


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
