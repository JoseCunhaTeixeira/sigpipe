"""Picking a curve by hand: at each frequency, the energy maximum inside a lasso drawn on the
dispersion image (PAC's lasso tool)."""

from collections.abc import Sequence

import numpy as np
from matplotlib.path import Path

from sigpipe.algorithms.picking.dispersion.curve import (
    clean_picks,
    lorentzian_uncertainty,
    resample_wavelength,
)
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.dispersion_image import DispersionImage


def pick_lasso(
    image: DispersionImage, polygon: Sequence[tuple[float, float]], mode: Mode
) -> DispersionImage:
    """`image` with the curve of `mode` picked inside `polygon`, its (frequency, velocity)
    vertices: at each frequency, the velocity of the energy maximum inside it; the curve
    cleaned (outliers replaced, then smoothed) and resampled over wavelength. It replaces the
    mode's curve, if `image` had one."""
    if len(polygon) < 3:
        raise ValueError("polygon must have at least 3 points")

    fs, vs, fv_map = image.fs, image.vs, image.fv_map

    poly = np.asarray(polygon, dtype=np.float64).copy()
    # Force the closing edge vertical at the starting frequency, so the implicit
    # segment from the last lasso point back to the first doesn't bias the mask
    # at the low-frequency end.
    poly[-1, 0] = poly[0, 0]

    f_indices = np.clip(np.searchsorted(fs, poly[:, 0]), 0, len(fs) - 1)
    v_indices = np.clip(np.searchsorted(vs, poly[:, 1]), 0, len(vs) - 1)
    f_start_i, f_end_i = int(f_indices.min()), int(f_indices.max())
    v_start_i, v_end_i = int(v_indices.min()), int(v_indices.max())

    if f_end_i <= f_start_i or v_end_i <= v_start_i + 1:
        raise ValueError("lasso selection is too small to pick a curve")

    F, V = np.meshgrid(fs, vs, indexing="ij")
    coords = np.column_stack([F.ravel(), V.ravel()])
    mask = Path(poly).contains_points(coords).reshape(fv_map.shape)
    fv_masked = np.where(mask, fv_map, 0.0)

    f_picked: list[float] = []
    v_picked: list[float] = []
    for row_i in range(f_start_i, f_end_i):
        window = fv_masked[row_i, v_start_i + 1 : v_end_i]
        if not np.any(window):
            continue
        true_v_i = v_start_i + 1 + int(np.argmax(window))
        # A pick sitting exactly on the window's upper edge is usually an
        # artifact of the mask boundary rather than a real spectral maximum.
        if true_v_i == v_end_i - 1 and v_picked:
            v_picked.append(v_picked[-1])
        else:
            v_picked.append(float(vs[true_v_i]))
        f_picked.append(float(fs[row_i]))

    # Drop the first row: it sits right where the lasso's vertical closing
    # edge was forced, so its pick is the least reliable.
    f_picked_arr = np.asarray(f_picked[1:], dtype=np.float32)
    if len(f_picked_arr) < 2:
        raise ValueError("lasso selection did not cover enough frequency rows to pick a curve")
    v_picked_arr = np.asarray(
        clean_picks(f_picked_arr, np.asarray(v_picked[1:]), polyorder=3), dtype=np.float32
    )

    new_curve = resample_wavelength(
        DispersionCurve(
            fs=f_picked_arr,
            vs=v_picked_arr,
            mode=mode,
            type=image.type,
            acquisition=image.acquisition,
            vs_err=lorentzian_uncertainty(f_picked_arr, v_picked_arr, image.acquisition),
        )
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
