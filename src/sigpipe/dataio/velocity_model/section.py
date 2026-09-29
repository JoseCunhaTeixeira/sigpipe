from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.base.velocity_model import VelocityModelsSection
from sigpipe.dataio.plot_config import CM, DISP_DPI, HEIGHT_CM, SINGLE_COLUMN_CM

_QUANTITY_LABELS = {
    "vs": ("Shear wave velocity $v_{S}$ [m/s]", 0),
    "vp": ("Compression wave velocity $v_{P}$ [m/s]", 1),
    "rho": ("Density [kg/m$^3$]", 2),
    "vs_std": ("Shear wave velocity std [m/s]", 3),
}


def save_velocity_models_sections(
    sections: dict[str, VelocityModelsSection],
    path: Path,
    dz: float | None = None,
    dx: float | None = None,
) -> None:
    """Save Vs(x, z) section grids for several model variants (e.g. best,
    median, ensemble, ...) into one HDF5 file, one group per variant.

    Each group holds that variant's own x positions, z elevations, and Vs(x, z)
    grid (NaN where a position's model doesn't reach that elevation) --
    variants are not forced onto a shared grid, since their depth extents can
    legitimately differ (e.g. a blocky "best" model vs. a finely-resampled
    "ensemble" model).
    """
    path = path.with_suffix(".hdf5")
    with h5py.File(path, "w") as file:
        for name, section in sections.items():
            xs, zs, vs_s_grid, _vs_p_grid, _rhos_grid, _vs_s_std_grid = section.to_grid(
                dz=dz, dx=dx
            )
            group = file.create_group(name)
            group.create_dataset("x", data=xs)
            group.create_dataset("z", data=zs)
            group.create_dataset("vs", data=vs_s_grid)


def load_velocity_models_sections(
    path: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load Vs(x, z) section grids previously written by
    `save_velocity_models_sections`.

    Returns a dict mapping each saved variant's name to its own (x, z, vs)
    arrays -- mirrors the per-variant, non-shared-grid layout of the save side.
    """
    with h5py.File(path, "r") as file:
        return {
            name: (group["x"][()], group["z"][()], group["vs"][()]) for name, group in file.items()
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
    Y-axis: elevation [m] (decreasing downward)
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

    ax.set_xlim(xs[0], xs[-1])

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(clabel)

    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Elevation [m]")
    fig.tight_layout()

    return fig
