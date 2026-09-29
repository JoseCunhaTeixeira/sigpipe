from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesSection
from sigpipe.dataio.plot_config import (
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

    ax.set_xlim(xs[0], xs[-1])

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(VELOCITY_TYPE_LABELS[pseudo_section.dispersion_curves[0].type])

    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Frequency [Hz]")
    fig.tight_layout()

    return fig


def pseudo_section_comparison_grids(
    observed: DispersionCurvesSection,
    predicted: DispersionCurvesSection,
    along: Literal["frequency", "wavelength"] = "frequency",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (positions, ys, obs_grid, pred_grid, residual) grids underlying
    a pseudo-section comparison, with no plotting -- for a JSON API and the
    figures a run saves (sigpipe.masw.inversion.section.save_comparison_figures).

    `observed` and `predicted` must cover the same positions (one curve per
    position in each, same mode). ys are frequencies, or with
    along="wavelength" wavelengths (each curve's velocity over its frequency),
    as many either way. obs_grid/pred_grid/residual have shape
    (n_positions, n_f). residual is (pred-obs)/pred*100.
    """
    positions = observed.xs
    if positions.shape != predicted.xs.shape or not np.allclose(positions, predicted.xs):
        raise ValueError("observed and predicted sections must cover the same positions")

    f_min = min(
        min(float(dc.fs.min()) for dc in observed),
        min(float(dc.fs.min()) for dc in predicted),
    )
    f_max = max(
        max(float(dc.fs.max()) for dc in observed),
        max(float(dc.fs.max()) for dc in predicted),
    )
    n_f = round(f_max - f_min) + 1

    def along_curve(dc: DispersionCurve) -> tuple[np.ndarray, np.ndarray]:
        """The curve's abscissae along the axis, increasing, and its velocities with them."""
        xs = dc.fs if along == "frequency" else dc.vs / dc.fs
        order = np.argsort(xs)
        return xs[order], dc.vs[order]

    curves = [along_curve(dc) for dc in (*observed, *predicted)]
    ys = np.linspace(
        min(float(xs.min()) for xs, _ in curves),
        max(float(xs.max()) for xs, _ in curves),
        n_f,
        dtype=np.float32,
    )

    pred_by_x = {float(dc.acquisition.xmid): dc for dc in predicted}

    obs_grid = np.full((len(positions), n_f), np.nan, dtype=np.float32)
    pred_grid = np.full((len(positions), n_f), np.nan, dtype=np.float32)
    for i, obs_curve in enumerate(observed):
        xs, vs = along_curve(obs_curve)
        mask = (ys >= xs.min()) & (ys <= xs.max())
        obs_grid[i, mask] = np.interp(ys[mask], xs, vs)

        xs, vs = along_curve(pred_by_x[float(obs_curve.acquisition.xmid)])
        mask_p = (ys >= xs.min()) & (ys <= xs.max())
        pred_grid[i, mask_p] = np.interp(ys[mask_p], xs, vs)

    if np.all(np.isnan(obs_grid)) or np.all(np.isnan(pred_grid)):
        residual = np.full_like(obs_grid, np.nan)
    else:
        residual = (pred_grid - obs_grid) / pred_grid * 100

    return positions, ys, obs_grid, pred_grid, residual
