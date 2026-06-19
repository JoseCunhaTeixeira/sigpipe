from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from sigproc.base.acquisition import Acquisition


@dataclass(slots=True, frozen=True)
class DispersionCurve:
    fs: np.ndarray
    vs: np.ndarray
    label: str
    type: str
    acquisitions: tuple[Acquisition, ...]
    vs_std: np.ndarray | None = None

    def __post_init__(self) -> None:
        fs = np.asarray(self.fs, dtype=np.float32)
        vs = np.asarray(self.vs, dtype=np.float32)

        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "vs", vs)

        self.fs.setflags(write=False)
        self.vs.setflags(write=False)

        if self.vs_std is not None:
            vs_std = np.asarray(self.vs_std, dtype=np.float32)
            object.__setattr__(self, "vs_std", vs_std)
            self.vs_std.setflags(write=False)


@dataclass(slots=True, frozen=True)
class DispersionCurves:
    curves: tuple[DispersionCurve, ...]

    def __iter__(self) -> Iterator[DispersionCurve]:
        return iter(self.curves)

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, item: int) -> DispersionCurve:
        return self.curves[item]


@dataclass(slots=True, frozen=True)
class DispersionImage:
    fv_map: np.ndarray
    fs: np.ndarray
    vs: np.ndarray
    type: str
    acquisitions: tuple[Acquisition, ...]
    dispersion_curves: DispersionCurves | None = None

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

    @property
    def has_curves(self) -> bool:
        return self.dispersion_curves is not None
