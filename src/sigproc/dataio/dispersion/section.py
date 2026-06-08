import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.sigproc.base.dispersion import DispersionCurves
from src.sigproc.dataio.plot_config import CM, DISP_DPI, HEIGHT_CM, SINGLE_COLUMN_CM


def plot_dispersion_curve_section(
    dispersion_curves: DispersionCurves,
) -> Figure:
    """
    Dispersion curve section.

    X-axis: curve index
    Y-axis: frequency [Hz]
    Color: phase velocity [m/s]
    """

    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )

    fmin = min(np.min(curve.fs) for curve in dispersion_curves)
    fmax = max(np.max(curve.fs) for curve in dispersion_curves)

    nfreq = 200
    fs_grid = np.linspace(fmin, fmax, nfreq)

    section = np.full(
        (nfreq, len(dispersion_curves)),
        np.nan,
        dtype=np.float32,
    )

    for i, curve in enumerate(dispersion_curves):
        mask = (fs_grid >= np.min(curve.fs)) & (fs_grid <= np.max(curve.fs))

        section[mask, i] = np.interp(
            fs_grid[mask],
            curve.fs,
            curve.vs,
        )

    pcm = ax.pcolormesh(
        np.arange(len(dispersion_curves)),
        fs_grid,
        section,
        shading="nearest",
        cmap="viridis",
    )

    cbar = fig.colorbar(pcm, ax=ax)
    vlabel = dispersion_curves[0].type.lower()
    clabel_map = {
        "phase": "Phase velocity [m/s]",
        "group": "Group velocity [m/s]",
        "": "Velocity [m/s]",
    }
    if vlabel not in clabel_map:
        raise ValueError(
            f"Unknown vlabel '{vlabel}'. Expected one of {tuple(clabel_map)}."
        )
    cbar.set_label(clabel_map[vlabel])

    ax.set_xlabel("Position [#]")
    ax.set_ylabel("Frequency [Hz]")

    return fig
