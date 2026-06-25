from dataclasses import dataclass

import numpy as np

from sigpipe.base.dispersion_curve import DispersionCurvesImage, VelocityType

from ._repr import array_repr
from .acquisition import Acquisition


@dataclass(slots=True, frozen=True)
class DispersionImage:
    fv_map: np.ndarray
    fs: np.ndarray
    vs: np.ndarray
    type: VelocityType
    acquisition: Acquisition
    dispersion_curves: DispersionCurvesImage | None = None

    def __post_init__(self) -> None:
        if self.fv_map.shape != (len(self.fs), len(self.vs)):
            raise ValueError(
                f"fv_map shape {self.fv_map.shape} does not match fs and vs lengths {(len(self.fs), len(self.vs))}"
            )

        object.__setattr__(self, "type", VelocityType(self.type))

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

    def __repr__(self) -> str:
        return (
            f"DispersionImage(fv_map={array_repr(self.fv_map)}, fs={array_repr(self.fs)}, "
            f"vs={array_repr(self.vs)}, type={self.type!r}, acquisition={self.acquisition!r}, "
            f"dispersion_curves={self.dispersion_curves!r})"
        )
