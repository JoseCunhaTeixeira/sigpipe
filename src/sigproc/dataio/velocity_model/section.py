from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sigproc.base.velocity_model import VelocityModelsSection
from sigproc.dataio.plot_config import CM, DISP_DPI, HEIGHT_CM, SINGLE_COLUMN_CM

_QUANTITY_LABELS = {
    "vs": ("Shear wave velocity $v_{S}$ [m/s]", 0),
    "vp": ("Compression wave velocity $v_{P}$ [m/s]", 1),
    "rho": ("Density [kg/m$^3$]", 2),
    "vs_std": ("Shear wave velocity std [m/s]", 3),
}


def plot_velocity_models_section(
    velocity_section: VelocityModelsSection,
    *,
    quantity: Literal["vs", "vp", "rho", "vs_std"] = "vs",
    dz: float = 0.01,
    dx: float | None = None,
) -> Figure:
    """
    Velocity section.

    X-axis: position [m]
    Y-axis: depth [m] (elevation, inverted)
    Color: requested quantity
    """
    clabel, grid_index = _QUANTITY_LABELS[quantity]

    xs, zs, *grids = velocity_section.to_grid(dz=dz, dx=dx)
    grid = grids[grid_index]

    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_CM * CM, HEIGHT_CM * CM),
        dpi=DISP_DPI,
    )

    pcm = ax.pcolormesh(xs, zs, grid.T, shading="nearest", cmap="viridis")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(clabel)

    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Elevation [m]")
    ax.invert_yaxis()
    fig.tight_layout()

    return fig
