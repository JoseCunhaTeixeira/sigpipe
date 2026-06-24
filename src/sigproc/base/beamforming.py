from dataclasses import dataclass

import numpy as np

from sigproc.base.acquisition import PlanarAcquisition

from ._repr import array_repr


@dataclass(slots=True, frozen=True)
class Beam:
    xy_map: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    acquisition: PlanarAcquisition

    def __post_init__(self) -> None:
        if not isinstance(self.acquisition, PlanarAcquisition):
            raise TypeError(
                "Beam requires a PlanarAcquisition (beamforming is over a scattered "
                f"(x, y) array, not a profiled line), got {type(self.acquisition).__name__}"
            )

        xy_map = np.asarray(self.xy_map, dtype=np.float32)
        xs = np.asarray(self.xs, dtype=np.float32)
        ys = np.asarray(self.ys, dtype=np.float32)

        object.__setattr__(self, "xy_map", xy_map)
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)

        self.xy_map.setflags(write=False)
        self.xs.setflags(write=False)
        self.ys.setflags(write=False)

    def __repr__(self) -> str:
        return (
            f"Beam(xy_map={array_repr(self.xy_map)}, xs={array_repr(self.xs)}, "
            f"ys={array_repr(self.ys)}, acquisition={self.acquisition!r})"
        )
