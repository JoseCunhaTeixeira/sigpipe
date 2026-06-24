import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from sigproc.base.dispersion_curve import DispersionCurvesSection
from sigproc.dataio.plot_config import (
    CM,
    DISP_DPI,
    DOUBLE_COLUMN_CM,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the (positions, fs, obs_grid, pred_grid, residual) grids underlying
    a pseudo-section comparison, with no plotting -- reusable for a JSON API
    as well as `plot_pseudo_section_comparison` below.

    `observed` and `predicted` must cover the same positions (one curve per
    position in each, same mode). obs_grid/pred_grid/residual have shape
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
    fs = np.linspace(f_min, f_max, n_f, dtype=np.float32)

    pred_by_x = {float(dc.acquisition.xmid): dc for dc in predicted}

    obs_grid = np.full((len(positions), n_f), np.nan, dtype=np.float32)
    pred_grid = np.full((len(positions), n_f), np.nan, dtype=np.float32)
    for i, obs_curve in enumerate(observed):
        mask = (fs >= obs_curve.fs.min()) & (fs <= obs_curve.fs.max())
        obs_grid[i, mask] = np.interp(fs[mask], obs_curve.fs, obs_curve.vs)

        pred_curve = pred_by_x[float(obs_curve.acquisition.xmid)]
        mask_p = (fs >= pred_curve.fs.min()) & (fs <= pred_curve.fs.max())
        pred_grid[i, mask_p] = np.interp(fs[mask_p], pred_curve.fs, pred_curve.vs)

    if np.all(np.isnan(obs_grid)) or np.all(np.isnan(pred_grid)):
        residual = np.full_like(obs_grid, np.nan)
    else:
        residual = (pred_grid - obs_grid) / pred_grid * 100

    return positions, fs, obs_grid, pred_grid, residual


def plot_pseudo_section_comparison(
    observed: DispersionCurvesSection,
    predicted: DispersionCurvesSection,
) -> Figure:
    """
    Observed vs. predicted dispersion-curve pseudo-sections, plus their residual.

    3 stacked panels: observed (viridis), predicted (viridis), residual as a
    percentage (bwr, symmetric).

    Port of the old Streamlit app's `display_pseudo_sections` in `display.py`.
    """
    positions, fs, obs_grid, pred_grid, residual = pseudo_section_comparison_grids(
        observed, predicted
    )

    finite_residual = residual[~np.isnan(residual)]
    lim = float(np.max(np.abs(finite_residual))) if finite_residual.size else 1.0

    fig, (ax_obs, ax_pred, ax_res) = plt.subplots(
        3, 1, figsize=(DOUBLE_COLUMN_CM * CM, 15 * CM), dpi=DISP_DPI, sharex=True
    )

    velocity_label = VELOCITY_TYPE_LABELS[observed.dispersion_curves[0].type]

    # Obs and pred share one color scale so the two panels are directly comparable.
    zmin = float(np.nanmin([np.nanmin(obs_grid), np.nanmin(pred_grid)]))
    zmax = float(np.nanmax([np.nanmax(obs_grid), np.nanmax(pred_grid)]))

    pcm_obs = ax_obs.pcolormesh(
        positions, fs, obs_grid.T, shading="nearest", cmap="viridis", vmin=zmin, vmax=zmax
    )
    fig.colorbar(pcm_obs, ax=ax_obs, label=f"Obs {velocity_label}")
    ax_obs.set_ylabel("Frequency [Hz]")

    pcm_pred = ax_pred.pcolormesh(
        positions, fs, pred_grid.T, shading="nearest", cmap="viridis", vmin=zmin, vmax=zmax
    )
    fig.colorbar(pcm_pred, ax=ax_pred, label=f"Pred {velocity_label}")
    ax_pred.set_ylabel("Frequency [Hz]")

    pcm_res = ax_res.pcolormesh(
        positions, fs, residual.T, shading="nearest", cmap="bwr", vmin=-lim, vmax=lim
    )
    fig.colorbar(pcm_res, ax=ax_res, label="Residuals [%]")
    ax_res.set_xlabel("Position [m]")
    ax_res.set_ylabel("Frequency [Hz]")

    ax_obs.set_xlim(positions[0], positions[-1])
    ax_pred.set_xlim(positions[0], positions[-1])
    ax_res.set_xlim(positions[0], positions[-1])

    fig.tight_layout()

    return fig
