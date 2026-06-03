from dataclasses import dataclass

import numpy as np

from src.base.acquisition import Acquisition


@dataclass(slots=True, frozen=True)
class DispersionCurve:
    fs: np.ndarray
    vs: np.ndarray
    name: str
    type: str
    acquisitions: tuple[Acquisition, ...]

    def __post_init__(self) -> None:
        fs = np.asarray(self.fs, dtype=np.float32)
        vs = np.asarray(self.vs, dtype=np.float32)

        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "vs", vs)

        self.fs.setflags(write=False)
        self.vs.setflags(write=False)


@dataclass(slots=True, frozen=True)
class DispersionImage:
    fv_map: np.ndarray
    fs: np.ndarray
    vs: np.ndarray
    type: str
    acquisitions: tuple[Acquisition, ...]
    dispersion_curves: tuple[DispersionCurve, ...] = ()

    def __post_init__(self) -> None:
        fv_map = np.asarray(self.fv_map, dtype=np.float32)
        fs = np.asarray(self.fs, dtype=np.float32)
        vs = np.asarray(self.vs, dtype=np.float32)

        object.__setattr__(self, "fv_map", fv_map)
        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "vs", vs)

        self.fv_map.setflags(write=False)
        self.fs.setflags(write=False)
        self.vs.setflags(write=False)
