from dataclasses import dataclass

import numpy as np

from sigproc.base.coordinate import UNKNOWN_COORDINATE, Coordinate


@dataclass(slots=True, frozen=True)
class Acquisition:
    source: Coordinate
    receivers: tuple[Coordinate, ...]

    def __eq__(self, other: object) -> bool:
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

    @property
    def offsets(self) -> np.ndarray:
        return np.array(
            [self.source.distance_to(r) for r in self.receivers],
            dtype=np.float32,
        )

    @property
    def middle_position(self) -> Coordinate:
        points = (self.source, *self.receivers)
        return Coordinate(
            x=(min(p.x for p in points) + max(p.x for p in points)) / 2,
            y=(min(p.y for p in points) + max(p.y for p in points)) / 2,
            z=(min(p.z for p in points) + max(p.z for p in points)) / 2,
        )

    @property
    def xmid(self) -> float:
        return (min(p.x for p in self.receivers) + max(p.x for p in self.receivers)) / 2


UNKNOWN_ACQUISITION = Acquisition(
    source=UNKNOWN_COORDINATE,
    receivers=(UNKNOWN_COORDINATE,),
)
