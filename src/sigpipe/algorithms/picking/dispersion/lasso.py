"""Picking a curve by hand: at each frequency, the energy maximum inside a lasso drawn on the
dispersion image (PAC's lasso tool). The picker's lasso (PickingParameters.lasso) is this one."""

from collections.abc import Sequence

import numpy as np
from matplotlib.path import Path

from sigpipe.algorithms.picking.dispersion.curve import (
    brightest_peaks,
    clean_picks,
    lorentzian_uncertainty,
    resample_runs,
)
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.dispersion_image import DispersionImage


def pick_lasso(
    image: DispersionImage, polygon: Sequence[tuple[float, float]], mode: Mode
) -> DispersionImage:
    """`image` with the curve of `mode` picked inside `polygon`, its (frequency, velocity)
    vertices, closed as drawn: at each frequency, the velocity of the brightest peak of the
    image's column inside it (brightest_peaks), none where the lasso's edge cuts a ridge still
    rising there; the curve cleaned (outliers replaced, then smoothed: clean_picks) and
    resampled over wavelength run by run (resample_runs: nothing made up across a gap in the
    picks), its points kept inside the lasso. It replaces the mode's curve, if `image` had
    one."""
    if len(polygon) < 3:
        raise ValueError("polygon must have at least 3 points")

    inside = lasso_cells(image, polygon)
    if np.count_nonzero(inside.any(axis=1)) < 2:
        raise ValueError("lasso selection is too small to pick a curve")

    rows, vs = lasso_picks(image, inside)
    if rows.size < 2:
        raise ValueError(
            "the image peaks inside the lasso at fewer than 2 frequencies, too few to pick a "
            "curve: draw it around a ridge"
        )
    new_curve = lasso_curve(image, rows, vs, polygon, mode)
    if new_curve is None:
        raise ValueError("too few of the curve's points stay inside the lasso to pick a curve")

    existing = [curve for curve in image.dispersion_curves or () if curve.mode != mode]
    return DispersionImage(
        fv_map=image.fv_map,
        fs=image.fs,
        vs=image.vs,
        type=image.type,
        acquisition=image.acquisition,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=(*existing, new_curve)),
    )


def lasso_cells(image: DispersionImage, polygon: Sequence[tuple[float, float]]) -> np.ndarray:
    """[n_f, n_v] mask: the cells of `image` inside `polygon`, its (frequency, velocity)
    vertices closed as drawn."""
    lasso = Path(np.asarray(polygon, dtype=np.float64))
    F, V = np.meshgrid(image.fs, image.vs, indexing="ij")
    return lasso.contains_points(np.column_stack([F.ravel(), V.ravel()])).reshape(F.shape)


def lasso_picks(
    image: DispersionImage, inside: np.ndarray, floor: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """The rows of `image` whose column peaks `inside` the lasso's cells (brightest_peaks) at
    `floor` or above, and their velocities, cleaned (clean_picks) from 2 rows on. A peak under
    the noise floor, where random phases sum, is noise."""
    rows, peaks = brightest_peaks(image.fv_map, image.vs, inside)
    if floor > 0 and rows.size:
        cells = np.clip(np.searchsorted(image.vs, peaks), 0, image.vs.size - 1)
        above = image.fv_map[rows, cells] >= floor
        rows, peaks = rows[above], peaks[above]
    if rows.size < 2:
        return rows, peaks.astype(np.float32)
    fs = image.fs[rows].astype(np.float32)
    return rows, np.asarray(clean_picks(fs, peaks), dtype=np.float32)


def lasso_curve(
    image: DispersionImage,
    rows: np.ndarray,
    velocities: np.ndarray,
    polygon: Sequence[tuple[float, float]],
    mode: Mode,
    step: float = 1.0,
) -> DispersionCurve | None:
    """The picks at `rows` as `mode`'s curve: with Lorentzian uncertainties, resampled over
    wavelength every `step` metres run by run (resample_runs), its points kept inside `polygon`;
    None under 2.

    Between two picks the resampling may step out of a lasso drawn close to the ridge, and the
    smoothing move a pick at its edge: the curve keeps the points inside the lasso."""
    fs = image.fs[rows].astype(np.float32)
    vs = np.asarray(velocities, dtype=np.float32)
    resampled = resample_runs(
        DispersionCurve(
            fs=fs,
            vs=vs,
            mode=mode,
            type=image.type,
            acquisition=image.acquisition,
            vs_err=lorentzian_uncertainty(fs, vs, image.acquisition),
        ),
        rows,
        step=step,
    )
    lasso = Path(np.asarray(polygon, dtype=np.float64))
    kept = lasso.contains_points(np.column_stack([resampled.fs, resampled.vs]))
    if np.count_nonzero(kept) < 2:
        return None
    return DispersionCurve(
        fs=resampled.fs[kept],
        vs=resampled.vs[kept],
        mode=resampled.mode,
        acquisition=resampled.acquisition,
        vs_err=resampled.vs_err[kept] if resampled.vs_err is not None else None,
        type=resampled.type,
    )
