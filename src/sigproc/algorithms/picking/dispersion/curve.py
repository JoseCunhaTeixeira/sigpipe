import numpy as np
from scipy.signal import medfilt, savgol_filter

from sigproc.base.dispersion import (
    DispersionCurve,
    DispersionCurves,
    DispersionImage,
)


def pick_curves(
    dispersion_image: DispersionImage,
    fmins: list[float] | list[None] = [None],
    fmaxs: list[float] | list[None] = [None],
    vmins: list[float] | list[None] = [None],
    vmaxs: list[float] | list[None] = [None],
    lbdmins: list[float] | list[None] = [None],
    lbdmaxs: list[float] | list[None] = [None],
    labels: list[str] = ["unknown"],
) -> DispersionImage:

    dispersion_curves: list[DispersionCurve] = (
        list(dispersion_image.dispersion_curves)
        if dispersion_image.dispersion_curves is not None
        else []
    )

    if (
        len(fmins)
        != len(fmaxs)
        != len(vmins)
        != len(vmaxs)
        != len(lbdmins)
        != len(lbdmaxs)
        != len(labels)
    ):
        raise ValueError(
            "requires same length for fmins, fmaxs, vmins, vmaxs, lbdmins, lbdmaxs, labels"
        )

    for fmin, fmax, vmin, vmax, lbdmin, lbdmax, label in zip(
        fmins, fmaxs, vmins, vmaxs, lbdmins, lbdmaxs, labels
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

        if (len(picked_vs) / 2) % 2 == 0:
            wl = len(picked_vs) // 2 + 1
        else:
            wl = len(picked_vs) // 2
        picked_vs = np.asarray(
            savgol_filter(picked_vs, window_length=wl, polyorder=2),
            dtype=np.float32,
        )

        dispersion_curves.append(
            DispersionCurve(
                fs=fs,
                vs=picked_vs,
                label=label,
                acquisitions=dispersion_image.acquisitions,
                type=dispersion_image.type,
            )
        )

    return DispersionImage(
        fv_map=dispersion_image.fv_map,
        fs=dispersion_image.fs,
        vs=dispersion_image.vs,
        type=dispersion_image.type,
        acquisitions=dispersion_image.acquisitions,
        dispersion_curves=DispersionCurves(curves=tuple(dispersion_curves)),
    )
