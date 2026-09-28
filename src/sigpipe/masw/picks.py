"""A window's picked curves, in PAC's layout: DispersionCurves_0000.csv beside the window's
dispersion image, whose figure is redrawn with them. Whoever picks (PAC's lasso or box, PACo's
automatic picker), a curve replaces the window's curve of the same mode and keeps the others."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.dataio.dispersion.loading import load_dispersion_curves
from sigpipe.dataio.dispersion.plotting import plot_dispersion_image
from sigpipe.dataio.dispersion.saving import save_dispersion_curves
from sigpipe.transformers import Plot

CURVES_FILE = "DispersionCurves_0000.csv"
FIGURE_FILE = "DispersionImage_0000.png"
PSEUDO_SECTION_POINTS = 200


def load_curves(folder: Path) -> DispersionCurvesImage | None:
    """The curves picked in window folder `folder`; None when it has none."""
    path = folder / CURVES_FILE
    if not path.exists():
        return None
    curves = tuple(load_dispersion_curves([path])[0])
    return DispersionCurvesImage(dispersion_curves=curves) if curves else None


def save_curves(folder: Path, image: DispersionImage, curves: DispersionCurvesImage | None) -> None:
    """`curves` saved as the picks of window folder `folder` (None: the file removed), and the
    window's figure of `image` redrawn with them."""
    path = folder / CURVES_FILE
    if curves:
        save_dispersion_curves(curves, path=path)
    else:
        path.unlink(missing_ok=True)
    figure = plot_dispersion_image(
        image,
        picked_curves=curves,
        # Where the checks' flags start: under lbmin the aliasing zone, over lbmax beyond the
        # window's reach.
        lbmin=min_resolvable_wavelength(image.acquisition),
        lbmax=longest_reached_wavelength(image.acquisition),
        normalize=True,
        show_errorbars=True,
    )
    Plot.savefig(path=folder / FIGURE_FILE, figure=figure)
    plt.close(figure)


def save_pick(folder: Path, image: DispersionImage, curve: DispersionCurve) -> bool:
    """`curve` saved in window folder `folder` in place of its curve of the same mode, the
    others kept, and the figure redrawn. Returns whether it replaced a curve."""
    saved = load_curves(folder)
    others = [one for one in saved or () if one.mode != curve.mode]
    save_curves(folder, image, DispersionCurvesImage(dispersion_curves=(*others, curve)))
    return saved is not None and len(others) < len(saved.dispersion_curves)


def remove_pick(folder: Path, image: DispersionImage, mode: Mode) -> DispersionCurvesImage | None:
    """The curve of `mode` removed from the picks of window folder `folder`, and the figure
    redrawn. Returns the curves left; raises ValueError when the window has no such curve."""
    saved = load_curves(folder)
    remaining = tuple(one for one in saved or () if one.mode != mode)
    if saved is None or len(remaining) == len(saved.dispersion_curves):
        raise ValueError(f"No curve labelled '{mode.label}' in {folder.name}")
    left = DispersionCurvesImage(dispersion_curves=remaining) if remaining else None
    save_curves(folder, image, left)
    return left


@dataclass(slots=True, frozen=True)
class PseudoSection:
    """A mode's picked curves along the line, on common grids: velocity by position and
    frequency, and by position and wavelength (NaN where a window's curve does not reach)."""

    positions: np.ndarray
    fs_grid: np.ndarray
    velocities_by_frequency: np.ndarray
    lambdas_grid: np.ndarray
    velocities_by_wavelength: np.ndarray


def pseudo_section(
    positions: Sequence[float],
    curves: Sequence[DispersionCurve | None],
    points: int = PSEUDO_SECTION_POINTS,
) -> PseudoSection:
    """The pseudo-section of `curves`, one per position (None where a window has none), on
    grids of `points` frequencies and wavelengths spanning them all."""
    picked = [curve for curve in curves if curve is not None]
    if not picked:
        raise ValueError("A pseudo-section needs at least one curve")
    fs_grid = np.linspace(
        min(float(curve.fs.min()) for curve in picked),
        max(float(curve.fs.max()) for curve in picked),
        points,
        dtype=np.float32,
    )
    by_frequency = np.full((len(positions), points), np.nan, dtype=np.float32)
    lambdas = [None if curve is None else curve.vs / curve.fs for curve in curves]
    lambdas_grid = np.linspace(
        min(float(lbd.min()) for lbd in lambdas if lbd is not None),
        max(float(lbd.max()) for lbd in lambdas if lbd is not None),
        points,
        dtype=np.float32,
    )
    by_wavelength = np.full((len(positions), points), np.nan, dtype=np.float32)
    for i, (curve, lbd) in enumerate(zip(curves, lambdas, strict=True)):
        if curve is None or lbd is None:
            continue
        mask = (fs_grid >= curve.fs.min()) & (fs_grid <= curve.fs.max())
        by_frequency[i, mask] = np.interp(fs_grid[mask], curve.fs, curve.vs)
        order = np.argsort(lbd)
        lbd_sorted, vs_sorted = lbd[order], curve.vs[order]
        mask = (lambdas_grid >= lbd_sorted.min()) & (lambdas_grid <= lbd_sorted.max())
        by_wavelength[i, mask] = np.interp(lambdas_grid[mask], lbd_sorted, vs_sorted)
    return PseudoSection(
        positions=np.asarray(positions, dtype=np.float32),
        fs_grid=fs_grid,
        velocities_by_frequency=by_frequency,
        lambdas_grid=lambdas_grid,
        velocities_by_wavelength=by_wavelength,
    )
