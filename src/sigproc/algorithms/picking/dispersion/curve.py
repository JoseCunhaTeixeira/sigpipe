import numpy as np
from scipy.signal import medfilt, savgol_filter

from sigproc.base.acquisition import Acquisition
from sigproc.base.dispersion_curve import (
    DispersionCurve,
    DispersionCurvesImage,
    Mode,
)
from sigproc.base.dispersion_image import DispersionImage


def lorentzian_uncertainty(
    fs: np.ndarray,
    vs: np.ndarray,
    acquisition: Acquisition,
    a: float = 0.5,
) -> np.ndarray:
    """Per-point phase-velocity uncertainty from the receiver array's resolving power.

    Lorentzian resolution formula: a tighter array (more receivers, smaller
    spacing) resolves velocity more precisely, so its curve gets a smaller
    uncertainty. Receiver count and spacing are taken from the curve's own
    acquisition geometry (assumed uniform, from the first two receivers).
    """

    receivers = acquisition.receivers
    n_receivers = len(receivers)
    dx = abs(receivers[1].x - receivers[0].x)

    fs = np.asarray(fs, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)

    fac = 10 ** (1 / np.sqrt(n_receivers * dx))
    dc_left = 1 / (1 / vs - 1 / (2 * fs * n_receivers * fac * dx))
    dc_right = 1 / (1 / vs + 1 / (2 * fs * n_receivers * fac * dx))
    resolution = np.abs(dc_left - dc_right)

    raw = (10**-a) * resolution
    uncertainty = np.where(raw > 0.4 * vs, 0.4 * vs, raw)
    uncertainty = np.where(raw < 5, 5, uncertainty)

    return uncertainty.astype(np.float32)


def pick_curves(
    dispersion_image: DispersionImage,
    fmins: list[float | None] | None = None,
    fmaxs: list[float | None] | None = None,
    vmins: list[float | None] | None = None,
    vmaxs: list[float | None] | None = None,
    lbdmins: list[float | None] | None = None,
    lbdmaxs: list[float | None] | None = None,
    labels: list[str] | None = None,
    modes: list[int] | None = None,
) -> DispersionImage:

    if fmins is None:
        fmins = [None]
    if fmaxs is None:
        fmaxs = [None]
    if vmins is None:
        vmins = [None]
    if vmaxs is None:
        vmaxs = [None]
    if lbdmins is None:
        lbdmins = [None]
    if lbdmaxs is None:
        lbdmaxs = [None]
    if labels is None:
        labels = [""]
    if modes is None:
        modes = list(range(len(labels)))

    dispersion_curves: list[DispersionCurve] = (
        list(dispersion_image.dispersion_curves)
        if dispersion_image.dispersion_curves is not None
        else []
    )

    lengths = {
        len(fmins),
        len(fmaxs),
        len(vmins),
        len(vmaxs),
        len(lbdmins),
        len(lbdmaxs),
        len(labels),
        len(modes),
    }
    if len(lengths) > 1:
        raise ValueError(
            "requires same length for fmins, fmaxs, vmins, vmaxs, lbdmins, lbdmaxs, labels, modes"
        )

    for fmin, fmax, vmin, vmax, lbdmin, lbdmax, label, mode in zip(
        fmins, fmaxs, vmins, vmaxs, lbdmins, lbdmaxs, labels, modes, strict=False
    ):
        fs = dispersion_image.fs.copy()
        vs = dispersion_image.vs.copy()
        fv_map = dispersion_image.fv_map.copy()

        F, V = np.meshgrid(fs, vs, indexing="ij")
        wavelength = V / (F + 1e-12)
        wavelength_mask = np.ones_like(fv_map, dtype=bool)
        if lbdmin is not None:
            wavelength_mask &= wavelength >= lbdmin
        if lbdmax is not None:
            wavelength_mask &= wavelength <= lbdmax
        fv_map[~wavelength_mask] = np.nan

        mask_f = np.ones_like(fs, dtype=bool)
        if fmin is not None:
            mask_f &= fs >= fmin
        if fmax is not None:
            mask_f &= fs <= fmax
        mask_v = np.ones_like(vs, dtype=bool)
        if vmin is not None:
            mask_v &= vs >= vmin
        if vmax is not None:
            mask_v &= vs <= vmax
        fs = fs[mask_f]
        vs = vs[mask_v]
        fv_map = fv_map[mask_f][:, mask_v]

        valid_rows = ~np.all(np.isnan(fv_map), axis=1)
        fs = fs[valid_rows]
        fv_map = fv_map[valid_rows]

        idx = np.array([np.where(row == np.nanmax(row))[0][-1] for row in fv_map])
        picked_vs = vs[idx]

        median_vs = medfilt(picked_vs, kernel_size=5)
        residual = np.abs(picked_vs - median_vs)
        threshold = 2.5 * np.median(residual)
        outliers = residual > threshold
        if np.any(outliers):
            valid = ~outliers
            picked_vs[outliers] = np.interp(
                fs[outliers],
                fs[valid],
                picked_vs[valid],
            )

        wl = len(picked_vs) // 2 + 1 if len(picked_vs) / 2 % 2 == 0 else len(picked_vs) // 2
        picked_vs = np.asarray(
            savgol_filter(picked_vs, window_length=wl, polyorder=2),
            dtype=np.float32,
        )

        dispersion_curves.append(
            DispersionCurve(
                fs=fs,
                vs=picked_vs,
                mode=Mode(label if label else "M", mode),
                acquisition=dispersion_image.acquisition,
                type=dispersion_image.type,
                vs_err=lorentzian_uncertainty(fs, picked_vs, dispersion_image.acquisition),
            )
        )

    return DispersionImage(
        fv_map=dispersion_image.fv_map,
        fs=dispersion_image.fs,
        vs=dispersion_image.vs,
        type=dispersion_image.type,
        acquisition=dispersion_image.acquisition,
        dispersion_curves=DispersionCurvesImage(dispersion_curves=tuple(dispersion_curves)),
    )
