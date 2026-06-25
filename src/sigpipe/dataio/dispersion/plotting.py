from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.base.dispersion_curve import DispersionCurves, VelocityType
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.dataio.plot_config import (
    CM,
    DISP_DPI,
    HEIGHT_CM,
    SINGLE_COLUMN_CM,
    VELOCITY_TYPE_LABELS,
)


class FrequencyUnity(StrEnum):
    HZ = "Hz"
    KHZ = "kHz"
    KHZMM = "kH.mm"


def plot_dispersion_image(
    dispersion_image: DispersionImage,
    *,
    picked_curves: DispersionCurves | None = None,
    modeled_curves: DispersionCurves | None = None,
    full_modeled_curves: DispersionCurves | None = None,
    thickness: float | None = None,
    lbmin: float | None = None,
    lbmax: float | None = None,
    normalize: bool = False,
    show_cbar: bool = False,
    show_legend: bool = False,
    show_errorbars: bool = False,
    unity: FrequencyUnity = FrequencyUnity.HZ,
) -> Figure:
    """
    Plot a frequency-velocity dispersion image.

    `picked_curves`/`modeled_curves` are drawn at their own frequencies
    (dots+line for picked, dots only for modeled). `full_modeled_curves` is
    for forward-modeled curves spanning the image's full frequency axis
    across every mode the model supports (drawn as a plain line, no markers)
    — e.g. to show all superior modes a model predicts, not just the picked
    ones.
    """
    _validate_bounds(lbmin, lbmax)
    fv_plot = _normalize_rows(dispersion_image.fv_map) if normalize else dispersion_image.fv_map
    fs_plot, xlabel = _scale_frequency(
        fs=dispersion_image.fs,
        thickness=thickness,
        unity=unity,
    )
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )
    mesh = ax.pcolormesh(
        fs_plot,
        dispersion_image.vs,
        fv_plot.T,
        cmap="gist_stern_r",
        shading="auto",
        vmin=0.0 if normalize else None,
        vmax=1.0 if normalize else None,
    )
    if show_cbar:
        fig.colorbar(mesh, ax=ax, pad=0.02, aspect=40)
    for bound in (lbmin, lbmax):
        if bound is not None:
            ax.plot(
                fs_plot,
                fs_plot * bound,
                color="grey",
                linestyle="--",
                linewidth=1.0,
                label="λ",
                zorder=1,
            )

    # Observed/picked curves always render behind modeled/full_modeled ones,
    # regardless of draw order, so they're never hidden by a curve drawn on
    # top of them.
    if picked_curves is not None:
        for picked_curve in picked_curves:
            picked_fs, _ = _scale_frequency(
                fs=picked_curve.fs,
                thickness=thickness,
                unity=unity,
            )
            ax.errorbar(
                picked_fs,
                picked_curve.vs,
                yerr=picked_curve.vs_err if show_errorbars else None,
                fmt="o-",
                color="white",
                ms=1 if show_errorbars else 1.5,
                linewidth=0.5,
                elinewidth=0.5,
                capsize=0,
                label=f"{picked_curve.mode.wave}{picked_curve.mode.number}",
                zorder=2,
            )
    else:
        if dispersion_image.dispersion_curves:
            for picked_curve in dispersion_image.dispersion_curves:
                picked_fs, _ = _scale_frequency(
                    fs=picked_curve.fs,
                    thickness=thickness,
                    unity=unity,
                )
                ax.errorbar(
                    picked_fs,
                    picked_curve.vs,
                    yerr=picked_curve.vs_err if show_errorbars else None,
                    fmt="o-",
                    color="white",
                    ms=1 if show_errorbars else 1.5,
                    linewidth=0.5,
                    elinewidth=0.5,
                    capsize=0,
                    label=f"{picked_curve.mode.wave}{picked_curve.mode.number}",
                    zorder=2,
                )

    if modeled_curves is not None:
        for modeled_curve in modeled_curves:
            model_fs, _ = _scale_frequency(
                fs=modeled_curve.fs,
                thickness=thickness,
                unity=unity,
            )
            ax.errorbar(
                model_fs,
                modeled_curve.vs,
                fmt="o",
                color="orange",
                ms=2,
                linewidth=1.0,
                elinewidth=0.5,
                label=f"{modeled_curve.mode.wave}{modeled_curve.mode.number}",
                zorder=3,
            )

    if full_modeled_curves is not None:
        for full_modeled_curve in full_modeled_curves:
            full_model_fs, _ = _scale_frequency(
                fs=full_modeled_curve.fs,
                thickness=thickness,
                unity=unity,
            )
            ax.plot(
                full_model_fs,
                full_modeled_curve.vs,
                color="orange",
                linewidth=1.0,
                zorder=4,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(VELOCITY_TYPE_LABELS[dispersion_image.type])
    ax.set_xlim(fs_plot[0], fs_plot[-1])
    ax.set_ylim(dispersion_image.vs[0], dispersion_image.vs[-1])
    if show_legend is True:
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return fig


def plot_dispersion_curves(
    picked_curves: DispersionCurves | None = None,
    *,
    modeled_curves: DispersionCurves | None = None,
    thickness: float | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_errorbars: bool = False,
    unity: FrequencyUnity = FrequencyUnity.HZ,
) -> Figure:
    """
    Plot dispersion curves.
    """
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )
    xlabel = "Frequency [kHz]"
    vtype = VelocityType.UNKNOWN
    if picked_curves is not None:
        cmap = plt.colormaps["viridis"]
        for i, picked_curve in enumerate(picked_curves):
            picked_fs, xlabel = _scale_frequency(
                fs=picked_curve.fs,
                thickness=thickness,
                unity=unity,
            )
            ax.errorbar(
                picked_fs,
                picked_curve.vs,
                yerr=picked_curve.vs_err if show_errorbars else None,
                fmt="-",
                color=cmap(i / max(len(picked_curves) - 1, 1)),
                linewidth=0.5,
                elinewidth=0.5,
                capsize=0,
                label=f"{picked_curve.mode.wave}{picked_curve.mode.number}",
                zorder=2,
            )
        vtype = picked_curves[0].type
    if modeled_curves is not None:
        for modeled_curve in modeled_curves:
            model_fs, xlabel = _scale_frequency(
                fs=modeled_curve.fs,
                thickness=thickness,
                unity=unity,
            )
            ax.plot(
                model_fs,
                modeled_curve.vs,
                color="red",
                linewidth=0.5,
                label=f"{modeled_curve.mode.wave}{modeled_curve.mode.number}",
                zorder=3,
            )
        vtype = modeled_curves[0].type
    ax.set_xlabel(xlabel)
    ax.set_ylabel(VELOCITY_TYPE_LABELS[vtype])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if fmin is not None and fmax is not None:
        ax.set_xlim(0.90 * fmin * 1e-3, 1.05 * fmax * 1e-3)
    if vmin is not None and vmax is not None:
        ax.set_ylim(0.90 * vmin, 1.05 * vmax)
    if ax.has_data():
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return fig


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    """Normalize each row independently to [0, 1]."""
    row_min = np.min(array, axis=1, keepdims=True)
    row_max = np.max(array, axis=1, keepdims=True)
    return np.asarray((array - row_min) / (row_max - row_min + 1e-12), dtype=array.dtype)


def _validate_bounds(
    lbmin: float | None,
    lbmax: float | None,
) -> None:
    """Validate lower-bound slope parameters."""
    if lbmin is not None and lbmin <= 0:
        raise ValueError(f"lbmin must be > 0, got {lbmin}")
    if lbmax is not None and lbmax <= 0:
        raise ValueError(f"lbmax must be > 0, got {lbmax}")
    if lbmin is not None and lbmax is not None and lbmin >= lbmax:
        raise ValueError(f"Expected lbmin < lbmax, got {lbmin} >= {lbmax}")


def _scale_frequency(
    fs: np.ndarray,
    unity: FrequencyUnity = FrequencyUnity.HZ,
    thickness: float | None = None,
) -> tuple[np.ndarray, str]:
    """Scale frequency axis and return axis label."""
    if unity == FrequencyUnity.HZ:
        return fs, "Frequency [Hz]"
    if unity == FrequencyUnity.KHZ:
        return fs * 1e-3, "Frequency [kHz]"
    if unity == FrequencyUnity.KHZMM and thickness is not None:
        return fs * thickness, "Frequency x Thickness [kHz.mm]"
    raise ValueError(f"unity and thickness combination not handled, got {unity} and {thickness}")
