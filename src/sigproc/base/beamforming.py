from dataclasses import dataclass

import numpy as np

from sigproc.base.acquisition import Acquisition


@dataclass(slots=True, frozen=True)
class Beam:
    xy_map: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    acquisition: Acquisition

    def __post_init__(self) -> None:
        xy_map = np.asarray(self.xy_map, dtype=np.float32)
        xs = np.asarray(self.xs, dtype=np.float32)
        ys = np.asarray(self.ys, dtype=np.float32)

        object.__setattr__(self, "xy_map", xy_map)
        object.__setattr__(self, "xs", xs)
        object.__setattr__(self, "ys", ys)

        self.xy_map.setflags(write=False)
        self.xs.setflags(write=False)
        self.ys.setflags(write=False)
