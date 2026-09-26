"""Lateral consistency: each window compared with its neighbours along the line, on either side
(PACo's line QC: G4 on the curves, G6 on the models, G8 on the petrophysical models): a window
fits a side when it lies within a misfit of the side's median; it is off a side only when that
side is full, agrees with itself and the window lies beyond. An isolated outlier is off
neighbours that agree and fits no side; a change shared with one side is geology."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

from sigpipe.base.dispersion_curve import DispersionCurve

type Standing = Literal["fits", "outlier", "shared_change", "no_neighbours"]


@dataclass(frozen=True)
class Series:
    """One window's values along an axis, sorted by it: a curve's velocities by wavelength (G4),
    a model's Vs by depth (G6)."""

    unit: str
    xmid: float
    x: np.ndarray  # increasing
    values: np.ndarray

    @classmethod
    def from_curve(cls, unit: str, xmid: float, curve: DispersionCurve) -> Self:
        """A dispersion curve as velocities by wavelength."""
        fs, vs = np.asarray(curve.fs, dtype=float), np.asarray(curve.vs, dtype=float)
        wavelengths = vs / fs
        order = np.argsort(wavelengths)
        return cls(unit, xmid, wavelengths[order], vs[order])

    def at(self, x: np.ndarray) -> np.ndarray:
        """The values at `x`, NaN outside the series' range."""
        inside = (x >= self.x[0]) & (x <= self.x[-1])
        return np.where(inside, np.interp(x, self.x, self.values), np.nan)


def median_series(series: Sequence[Series]) -> tuple[np.ndarray, np.ndarray]:
    """The median of `series` at the union of their x: (x, values)."""
    grid = np.unique(np.concatenate([one.x for one in series]))
    stack = np.array([one.at(grid) for one in series])
    return grid, np.nanmedian(stack, axis=0)


def misfit(
    series: Series, x: np.ndarray, median: np.ndarray, relative: bool = True
) -> tuple[float, int]:
    """The series' median misfit to `median` over the x both hold, relative to the median or in
    the values' own unit, and how many they share (the misfit is NaN when none)."""
    own = series.at(x)
    shared = ~np.isnan(own) & ~np.isnan(median)
    if not shared.any():
        return float("nan"), 0
    differences = np.abs(own[shared] - median[shared])
    if relative:
        differences = differences / median[shared]
    return float(np.median(differences)), int(shared.sum())


@dataclass(frozen=True)
class SideFit:
    """How a window sits against its neighbours on one side."""

    series: tuple[Series, ...]
    misfit: float  # to the side's median; NaN when nothing is shared
    shared: int  # x shared with the side's median
    compared: bool  # enough shared x to say anything
    fits: bool  # compared, and within max_misfit of the side's median
    off: bool  # compared, beyond max_misfit, and the side full and agreeing with itself


def fit_side(
    series: Series,
    side: Sequence[Series],
    max_misfit: float,
    min_shared: int,
    full: int,
    relative: bool = True,
) -> SideFit:
    """`series` against the neighbours `side`. It fits the side when it lies within `max_misfit`
    of their median; it is off the side only when the side is `full` (a side of one window
    agrees with itself), its windows agree with each other (each within half of `max_misfit` of
    their median) and the series lies beyond `max_misfit`. Misfits are relative, or in the
    values' own unit."""
    grid, median = median_series(side)
    agree = all(misfit(other, grid, median, relative)[0] <= max_misfit / 2 for other in side)
    value, shared = misfit(series, grid, median, relative)
    compared = shared >= min_shared
    fits = compared and value <= max_misfit
    off = compared and len(side) >= full and agree and value > max_misfit
    return SideFit(tuple(side), value, shared, compared, fits, off)


@dataclass(frozen=True)
class Neighbourhood:
    """A window against both its sides."""

    sides: tuple[SideFit, ...]  # the sides it has: one at the line's ends
    standing: Standing

    @property
    def compared(self) -> tuple[SideFit, ...]:
        return tuple(side for side in self.sides if side.compared)

    @property
    def off(self) -> tuple[SideFit, ...]:
        return tuple(side for side in self.sides if side.off)

    @property
    def worst(self) -> float:
        """The largest misfit to a side it was compared with; NaN when none."""
        return max((side.misfit for side in self.compared), default=float("nan"))

    @property
    def where(self) -> str:
        """Where an outlier is off, in words."""
        if len(self.off) == 2:
            return "on both sides"
        return (
            "on its only side" if len(self.sides) == 1 else "on one side, without fitting the other"
        )


def neighbourhoods(
    ordered: Sequence[Series],
    neighbours: int,
    max_misfit: float,
    min_shared: int,
    relative: bool = True,
    max_distance: float | None = None,
) -> tuple[Neighbourhood, ...]:
    """Each of `ordered` (sorted by xmid) against `neighbours` / 2 windows on each side, those
    within `max_distance` of it when given (a line with gaps: windows far away are not
    neighbours)."""
    per_side = max(1, neighbours // 2)
    found: list[Neighbourhood] = []
    for index, series in enumerate(ordered):
        left = ordered[max(0, index - per_side) : index]
        right = ordered[index + 1 : index + 1 + per_side]
        if max_distance is not None:
            left = [one for one in left if series.xmid - one.xmid <= max_distance]
            right = [one for one in right if one.xmid - series.xmid <= max_distance]
        sides = tuple(
            fit_side(series, side, max_misfit, min_shared, per_side, relative)
            for side in (left, right)
            if side
        )
        fits = any(side.fits for side in sides)
        off = any(side.off for side in sides)
        standing: Standing = (
            "outlier"
            if off and not fits
            else "shared_change"
            if off
            else "no_neighbours"
            if not fits
            else "fits"
        )
        found.append(Neighbourhood(sides, standing))
    return tuple(found)


def spread(values: Sequence[float]) -> float:
    """MAD over median: how much `values` vary along the line (0 for fewer than two)."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return 0.0
    return float(np.median(np.abs(array - np.median(array))) / np.median(array))
