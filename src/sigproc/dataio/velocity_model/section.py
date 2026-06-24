import warnings
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.lib.stride_tricks import sliding_window_view

from sigproc.base.velocity_model import VelocityModelsSection
from sigproc.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM, HEIGHT_CM, SINGLE_COLUMN_CM

_QUANTITY_LABELS = {
    "vs": ("Shear wave velocity $v_{S}$ [m/s]", 0),
    "vp": ("Compression wave velocity $v_{P}$ [m/s]", 1),
    "rho": ("Density [kg/m$^3$]", 2),
    "vs_std": ("Shear wave velocity std [m/s]", 3),
}


def _nanmedian_filter_axis0(grid: np.ndarray, size: int) -> np.ndarray:
    """nanmedian filter along axis 0 only, window width `size`.

    Vectorized equivalent of
    `scipy.ndimage.generic_filter(grid, nanmedian, size=(size, 1))` (same
    default edge-replicating "reflect" boundary, called "symmetric" in
    np.pad) -- generic_filter calls its Python callback once per output
    pixel, which is far too slow for realistic section grids (seconds for a
    single smoothing pass). This computes the same result in one vectorized
    nanmedian call instead.
    """
    left = size // 2
    right = size - 1 - left
    padded = np.pad(grid, ((left, right), (0, 0)), mode="symmetric")
    windows = sliding_window_view(padded, size, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result: np.ndarray = np.nanmedian(windows, axis=-1)
        return result


def smooth_laterally(grid: np.ndarray) -> np.ndarray:
    """Smooth a (positions, depths) section grid across positions.

    Three passes of a median filter with shrinking windows (4, 3, 2) along
    the position axis only -- depths are untouched. Port of the old
    Streamlit app's `mode_filter_median` + `generic_filter` cascade.
    """
    smoothed = grid
    for size in (4, 3, 2):
        smoothed = _nanmedian_filter_axis0(smoothed, size)
    return smoothed


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


def plot_velocity_and_std_section(
    velocity_section: VelocityModelsSection,
    *,
    dz: float = 0.01,
    dx: float | None = None,
    lateral_smoothing: bool = False,
) -> Figure:
    """
    Vs(x, z) and Vs std(x, z) sections, stacked.

    X-axis: position [m]
    Y-axis: elevation [m] (decreasing downward)
    Top: Vs, terrain colormap. Bottom: Vs std, afmhot_r colormap (vmin=0).

    Port of the old Streamlit app's `display_inverted_section` in `display.py`.
    """
    xs, zs, vs_s_grid, _vs_p_grid, _rhos_grid, vs_s_std_grid = velocity_section.to_grid(
        dz=dz, dx=dx
    )

    if lateral_smoothing:
        vs_s_grid = smooth_laterally(vs_s_grid)
        vs_s_std_grid = smooth_laterally(vs_s_std_grid)

    fig, (ax_vs, ax_std) = plt.subplots(
        2, 1, figsize=(DOUBLE_COLUMN_CM * CM, 10 * CM), dpi=DISP_DPI, sharex=True
    )

    pcm_vs = ax_vs.pcolormesh(xs, zs, vs_s_grid.T, shading="nearest", cmap="terrain")
    fig.colorbar(pcm_vs, ax=ax_vs, label="$v_{S}$ [m/s]")
    ax_vs.set_ylabel("Elevation [m]")

    pcm_std = ax_std.pcolormesh(xs, zs, vs_s_std_grid.T, shading="nearest", cmap="afmhot_r", vmin=0)
    fig.colorbar(pcm_std, ax=ax_std, label="Std [m/s]")
    ax_std.set_xlabel("Position [m]")
    ax_std.set_ylabel("Elevation [m]")

    ax_vs.set_xlim(xs[0], xs[-1])
    ax_std.set_xlim(xs[0], xs[-1])

    fig.tight_layout()

    return fig
