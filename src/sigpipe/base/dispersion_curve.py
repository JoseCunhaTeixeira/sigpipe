from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from ._repr import array_repr
from .acquisition import Acquisition


class VelocityType(StrEnum):
    UNKNOWN = ""
    PHASE = "phase"
    GROUP = "group"


class Mode(NamedTuple):
    """Wave type (e.g. "R" for Rayleigh, "L" for Love, "SH") and mode number."""

    wave: str
    number: int


@dataclass(slots=True, frozen=True)
class DispersionCurve:
    fs: np.ndarray
    vs: np.ndarray
    mode: Mode
    acquisition: Acquisition
    vs_err: np.ndarray | None = None
    type: VelocityType = VelocityType.UNKNOWN

    def __post_init__(self) -> None:
        if len(self.fs) != len(self.vs):
            raise ValueError(
                f"fs and vs arrays must have the same length, got {len(self.fs)} and {len(self.vs)}"
            )

        mode = (
            self.mode if isinstance(self.mode, Mode) else Mode(*self.mode)  # pyright: ignore[reportGeneralTypeIssues]
        )
        object.__setattr__(self, "mode", mode)

        if not mode.wave:
            raise ValueError("Mode wave type must be a non-empty string")
        if mode.number < 0:
            raise ValueError(f"Mode number must be non-negative, got {mode.number}")

        object.__setattr__(self, "type", VelocityType(self.type))

        if self.vs_err is not None:
            vs_err = np.asarray(self.vs_err, dtype=np.float32)
            object.__setattr__(self, "vs_err", vs_err)
            self.vs_err.setflags(write=False)

        fs = np.asarray(self.fs, dtype=np.float32)
        vs = np.asarray(self.vs, dtype=np.float32)

        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "vs", vs)

        self.fs.setflags(write=False)
        self.vs.setflags(write=False)

    def __repr__(self) -> str:
        return (
            f"DispersionCurve(fs={array_repr(self.fs)}, vs={array_repr(self.vs)}, "
            f"mode={self.mode!r}, acquisition={self.acquisition!r}, "
            f"vs_err={array_repr(self.vs_err)}, type={self.type!r})"
        )


@dataclass(slots=True, frozen=True)
class DispersionCurves:
    dispersion_curves: tuple[DispersionCurve, ...]

    def __post_init__(self) -> None:
        if len(self.dispersion_curves) == 0:
            raise ValueError("At least one dispersion curve is required")

        if not all(isinstance(curve, DispersionCurve) for curve in self.dispersion_curves):
            raise TypeError(
                "All elements of dispersion_curves must be instances of DispersionCurve"
            )

    def __iter__(self) -> Iterator[DispersionCurve]:
        return iter(self.dispersion_curves)

    def __len__(self) -> int:
        return len(self.dispersion_curves)

    def __getitem__(self, item: int) -> DispersionCurve:
        return self.dispersion_curves[item]


@dataclass(slots=True, frozen=True)
class DispersionCurvesImage(DispersionCurves):
    """
    Dispersion curves extracted from a dispersion image. The modes should be unique.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        modes = [dispersion_curve.mode for dispersion_curve in self.dispersion_curves]
        if len(modes) != len(set(modes)):
            raise ValueError(f"Duplicate modes found in dispersion curves: {modes}")

        # Sort by mode name and number
        ordered = tuple(
            sorted(
                self.dispersion_curves,
                key=lambda dc: (dc.mode.wave, dc.mode.number),
            )
        )

        object.__setattr__(self, "dispersion_curves", ordered)


@dataclass(slots=True, frozen=True)
class DispersionCurvesSection(DispersionCurves):
    """
    Dispersion curves as a pseudo section. All modes should be the same, each with an unique x coordinate.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        modes = {dc.mode for dc in self.dispersion_curves}
        if len(modes) != 1:
            raise ValueError(
                "All dispersion curves in a DispersionCurvesSection must have the same mode"
            )

        xs = [dc.acquisition.xmid for dc in self.dispersion_curves]
        if len(xs) != len(set(xs)):
            raise ValueError("All dispersion curves must have unique x coordinate")

        ordered = tuple(
            sorted(
                self.dispersion_curves,
                key=lambda dc: dc.acquisition.xmid,
            )
        )
        object.__setattr__(self, "dispersion_curves", ordered)

    @property
    def xs(self) -> np.ndarray:
        return np.array(
            [dc.acquisition.xmid for dc in self.dispersion_curves],
            dtype=np.float32,
        )

    def to_grid(
        self,
        dx: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        profile_xs = np.array(
            [dc.acquisition.xmid for dc in self.dispersion_curves],
            dtype=np.float32,
        )

        print(profile_xs)

        if dx is None:
            if len(profile_xs) < 2:
                raise ValueError("dx must be given when there is only one curve")
            dx = float(np.min(np.diff(profile_xs))) / 100

        f_top = min(float(dc.fs.max()) for dc in self.dispersion_curves)
        f_bottom = max(float(dc.fs.min()) for dc in self.dispersion_curves)

        df = min(np.min(np.diff(dc.fs)) for dc in self.dispersion_curves)

        if f_top <= f_bottom:
            raise ValueError("No common frequency range among dispersion curves")

        nf = int(np.floor((f_top - f_bottom) / df)) + 1

        fs = (f_top - np.arange(nf, dtype=np.float32) * df).astype(np.float32)

        fs_asc = fs[::-1]

        per_curve_vs = np.array(
            [np.interp(fs_asc, dc.fs, dc.vs)[::-1] for dc in self.dispersion_curves],
            dtype=np.float32,
        )

        nx = int(np.floor((profile_xs.max() - profile_xs.min()) / dx)) + 1

        xs = (profile_xs.min() + np.arange(nx, dtype=np.float32) * dx).astype(np.float32)

        nearest = np.abs(xs[:, None] - profile_xs[None, :]).argmin(axis=1)

        vs_grid = per_curve_vs[nearest]

        return xs, fs, vs_grid
