import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sigproc.base.dispersion_curve import DispersionCurvesSection
from sigproc.dataio.plot_config import (
    CM,
    DISP_DPI,
    HEIGHT_CM,
    SINGLE_COLUMN_CM,
    VELOCITY_TYPE_LABELS,
)


def plot_dispersion_curves_section(
    pseudo_section: DispersionCurvesSection,
    dx: float | None = None,
) -> Figure:
    """
    Velocity pseudo-section, built directly from dispersion curves (no inversion).

    X-axis: position [m]
    Y-axis: frequency [Hz]
    Color: phase/group velocity [m/s]
    """
    xs, fs, vs_grid = pseudo_section.to_grid(dx=dx)
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )
    pcm = ax.pcolormesh(xs, fs, vs_grid, shading="nearest", cmap="viridis")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(VELOCITY_TYPE_LABELS[pseudo_section.dispersion_curves[0].type])

    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Frequency [Hz]")
    fig.tight_layout()

    return fig
