import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from sigproc.base.coordinate import UNKNOWN_COORDINATE, Coordinate


@dataclass(slots=True, frozen=True)
class Acquisition:
    """Source/receiver positions with no assumption about their geometry.

    Offsets and a representative "middle" position depend on whether the
    receivers are scattered over a plane or strung along a profiled line,
    so the base class can't compute either without guessing -- use
    PlanarAcquisition or LinearAcquisition for that.
    """

    source: Coordinate
    receivers: tuple[Coordinate, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Acquisition):
            return NotImplemented

        return self.source == other.source and self.receivers == other.receivers

    @property
    def is_unknown(self) -> bool:
        return self.source.is_unknown or any(r.is_unknown for r in self.receivers)

    @property
    def offsets(self) -> np.ndarray:
        """Source-to-receiver distances.

        Not implemented on the base class: whether this is a straight
        chord or a distance along a topographic profile depends on the
        geometry, which only a concrete subclass knows.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement offsets; "
            "use PlanarAcquisition or LinearAcquisition"
        )

    @property
    def mid_position(self) -> Coordinate:
        """Representative 'middle' position of this acquisition.

        Not implemented on the base class: a centroid is right for a
        scattered array but wrong for a profiled line on irregular
        topography, and only a concrete subclass knows which applies.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement mid_position; "
            "use PlanarAcquisition or LinearAcquisition"
        )

    @property
    def xmid(self) -> float:
        return self.mid_position.x


@dataclass(slots=True, frozen=True)
class PlanarAcquisition(Acquisition):
    """An Acquisition scattered over a 2D (x, y) plane with no meaningful
    elevation/topography -- e.g. a passive beamforming array.

    offsets is the straight-line source-to-receiver chord and mid_position
    is the centroid of source and receivers together, correct when there's
    no implied travel path or ordering between them.
    """

    def __post_init__(self) -> None:
        # Source must be known too, not just receivers: offsets and
        # mid_position both fold the source into their math, so an unknown
        # (NaN) source would silently poison every result instead of
        # failing where the bad input actually is.
        if any(p.is_unknown for p in (self.source, *self.receivers)):
            raise ValueError(
                "PlanarAcquisition requires the source and all receivers to have known positions"
            )

        zs = [p.z for p in (self.source, *self.receivers)]
        if not np.allclose(zs, zs[0]):
            raise ValueError(
                "PlanarAcquisition requires all z coordinates to be equal "
                f"(no topography), got {sorted(set(zs))}"
            )

    @property
    def offsets(self) -> np.ndarray:
        return np.array(
            [self.source.distance_to(r) for r in self.receivers],
            dtype=np.float32,
        )

    @property
    def mid_position(self) -> Coordinate:
        """Centroid of source and receivers together.

        Unlike a linear survey's receiver line -- which the source sits
        outside of by construction -- a planar acquisition's source is
        just another point in the same scattered footprint, so it belongs
        in the center of mass rather than being excluded from it.
        """
        coordinates = [self.source, *self.receivers]
        return Coordinate(
            x=float(np.mean([r.x for r in coordinates])),
            y=float(np.mean([r.y for r in coordinates])),
            z=float(np.mean([r.z for r in coordinates])),
        )


@dataclass(slots=True, frozen=True)
class LinearAcquisition(Acquisition):
    """An Acquisition whose receivers (and source) lie along a single
    profiled line in (x, z) -- the standard MASW/refraction survey layout.

    offsets and mid_position follow the ground-surface arc rather than
    straight chords/centroid, since that's the path a surface wave (and
    the physical receiver line) actually takes over real topography.
    """

    def __post_init__(self) -> None:
        # Source must be known too, not just receivers: the arc-length walk
        # below includes the source, so an unknown (NaN) source would
        # silently poison every offset/mid_position result instead of
        # failing where the bad input actually is.
        if any(p.is_unknown for p in (self.source, *self.receivers)):
            raise ValueError(
                "LinearAcquisition requires the source and all receivers to have known positions"
            )

        ys = [p.y for p in (self.source, *self.receivers)]
        if not np.allclose(ys, ys[0]):
            raise ValueError(
                "LinearAcquisition requires all y coordinates to be equal "
                f"(needs x and z as topography), got {sorted(set(ys))}"
            )

    @property
    def offsets(self) -> np.ndarray:
        """Source-to-receiver distances along the (x, z) ground-surface
        profile through the receivers and source (ordered by x), not as
        straight chords -- the physically correct travel distance for a
        surface wave on irregular topography.
        """
        points = (self.source, *self.receivers)
        order = sorted(range(len(points)), key=lambda i: points[i].x)
        ordered = [points[i] for i in order]

        cumulative = [0.0]
        for a, b in pairwise(ordered):
            cumulative.append(cumulative[-1] + math.hypot(b.x - a.x, b.z - a.z))

        arc_length = [0.0] * len(points)
        for sorted_pos, original_idx in enumerate(order):
            arc_length[original_idx] = cumulative[sorted_pos]

        source_arc = arc_length[0]
        receiver_arcs = np.array(arc_length[1:], dtype=np.float32)
        result: np.ndarray = np.abs(receiver_arcs - source_arc)
        return result

    @property
    def mid_position(self) -> Coordinate:
        """Point at the geometric middle of the receiver line, located by
        arc length along the (x, z) ground-surface profile.

        A centroid or min/max average of the receivers only gives the
        right answer on flat or mildly-sloped ground -- on irregular
        topography the highest and lowest points in a window are rarely
        at its middle, so those averages can land far from the receiver
        line entirely. Walking the arc length instead finds the point
        that actually splits the receiver line in half, the same way
        PAC's window-naming midpoint does for x alone.

        The source is deliberately excluded, unlike PlanarAcquisition's
        centroid: PAC's window-building keeps the source outside the
        receiver line by construction, so folding it in here would pull
        the midpoint off the line instead of finding its actual middle.
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


UNKNOWN_ACQUISITION = Acquisition(
    source=UNKNOWN_COORDINATE,
    receivers=(UNKNOWN_COORDINATE,),
)

_ACQUISITION_KINDS: dict[str, type[Acquisition]] = {
    "Acquisition": Acquisition,
    "PlanarAcquisition": PlanarAcquisition,
    "LinearAcquisition": LinearAcquisition,
}


def acquisition_kind(acquisition: Acquisition) -> str:
    """Name identifying `acquisition`'s concrete geometry, to persist
    alongside its source/receivers so a loader can reconstruct the same
    subclass later -- the saved positions alone don't say whether they're
    a scattered plane or a profiled line.
    """
    return type(acquisition).__name__


def acquisition_from_kind(
    kind: str,
    source: Coordinate,
    receivers: tuple[Coordinate, ...],
) -> Acquisition:
    """Reconstruct the Acquisition subclass named by `kind` (as produced
    by `acquisition_kind`). Falls back to the geometry-agnostic base class
    for a missing or unrecognized kind (e.g. a file saved before this
    tag existed) rather than guessing a specific one.
    """
    cls = _ACQUISITION_KINDS.get(kind, Acquisition)
    return cls(source=source, receivers=receivers)
