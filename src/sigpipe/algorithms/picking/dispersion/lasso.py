"""Picking a curve by hand: at each frequency, the energy maximum inside a lasso drawn on the
dispersion image (PAC's lasso tool)."""

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

    lasso = Path(np.asarray(polygon, dtype=np.float64))
    F, V = np.meshgrid(image.fs, image.vs, indexing="ij")
    inside = lasso.contains_points(np.column_stack([F.ravel(), V.ravel()])).reshape(F.shape)
    if np.count_nonzero(inside.any(axis=1)) < 2:
        raise ValueError("lasso selection is too small to pick a curve")

    rows, peaks = brightest_peaks(image.fv_map, image.vs, inside)
    if rows.size < 2:
        raise ValueError(
            "the image peaks inside the lasso at fewer than 2 frequencies, too few to pick a "
            "curve: draw it around a ridge"
        )
    fs = image.fs[rows].astype(np.float32)
    vs = np.asarray(clean_picks(fs, peaks), dtype=np.float32)

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
    )
    # Between two picks the resampling may step out of a lasso drawn close to the ridge, and the
    # smoothing move a pick at its edge: the curve keeps the points inside the lasso.
    kept = lasso.contains_points(np.column_stack([resampled.fs, resampled.vs]))
    if np.count_nonzero(kept) < 2:
        raise ValueError("too few of the curve's points stay inside the lasso to pick a curve")
    new_curve = DispersionCurve(
        fs=resampled.fs[kept],
        vs=resampled.vs[kept],
        mode=resampled.mode,
        acquisition=resampled.acquisition,
        vs_err=resampled.vs_err[kept] if resampled.vs_err is not None else None,
        type=resampled.type,
    )

    existing = [curve for curve in image.dispersion_curves or () if curve.mode != mode]
    return DispersionImage(
        fv_map=image.fv_map,
        fs=image.fs,
        vs=image.vs,
        type=image.type,
        acquisition=image.acquisition,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=(*existing, new_curve)),
    )
