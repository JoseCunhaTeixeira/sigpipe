import math
from dataclasses import dataclass
from itertools import pairwise

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
    def arc_midpoint(self) -> Coordinate:
        """Point at the geometric middle of the receiver line, located by
        arc length along the (x, z) ground-surface profile.

        A min/max average of all receiver (and source) coordinates only
        gives the right answer on flat or mildly-sloped ground -- on
        irregular topography the highest and lowest points in a window
        are rarely at its middle, so that average can land far from the
        receiver line entirely. Walking the arc length instead finds the
        point that actually splits the receiver line in half, the same
        way PAC's window-naming midpoint does for x alone.
        """
        receivers = self.receivers
        if len(receivers) == 1:
            return receivers[0]

        cumulative = [0.0]
        for a, b in pairwise(receivers):
            cumulative.append(cumulative[-1] + math.hypot(b.x - a.x, b.z - a.z))

        half = cumulative[-1] / 2
        for i in range(1, len(cumulative)):
            if cumulative[i] >= half:
                segment = cumulative[i] - cumulative[i - 1]
                t = (half - cumulative[i - 1]) / segment if segment > 0 else 0.0
                a, b = receivers[i - 1], receivers[i]
                return Coordinate(
                    x=a.x + t * (b.x - a.x),
                    y=a.y + t * (b.y - a.y),
                    z=a.z + t * (b.z - a.z),
                )

        return receivers[-1]

    @property
    def xmid(self) -> float:
        return self.arc_midpoint.x


UNKNOWN_ACQUISITION = Acquisition(
    source=UNKNOWN_COORDINATE,
    receivers=(UNKNOWN_COORDINATE,),
)
