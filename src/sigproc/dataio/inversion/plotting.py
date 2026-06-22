import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from sigproc.base.inversion import InversionResult
from sigproc.dataio.plot_config import CM
from sigproc.dataio.velocity_model.plotting import plot_velocity_models


def _hdi_levels(density: np.ndarray, probs: tuple[float, ...]) -> list[float]:
    """Density thresholds whose enclosed area contains each given probability mass."""
    flat = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    levels = [flat[min(int(np.searchsorted(cumulative, p)), len(flat) - 1)] for p in probs]
    return sorted(levels)


def plot_posterior_marginals(
    samples: dict[str, np.ndarray],
    *,
    hdi_probs: tuple[float, ...] = (0.3, 0.6, 0.9),
    n_grid: int = 80,
    textsize: float = 9,
) -> Figure:
    """
    Corner plot of posterior samples.

    Diagonal: 1D KDE marginal for each parameter.
    Lower triangle: 2D KDE with highest-density-interval contours (hdi_probs).
    """
    names = list(samples.keys())
    n = len(names)

    fig, axs = plt.subplots(n, n, figsize=(2.6 * CM * n, 2.4 * CM * n), dpi=150, squeeze=False)

    grids_1d = {
        name: np.linspace(values.min(), values.max(), n_grid) for name, values in samples.items()
    }

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            ax = axs[i, j]

            if j > i:
                ax.axis("off")
                continue

            if i == j:
                kde = gaussian_kde(samples[name_i])
                xs = grids_1d[name_i]
                ax.plot(xs, kde(xs), color="tab:blue", linewidth=1)
                ax.set_yticks([])
            else:
                kde = gaussian_kde(np.vstack([samples[name_j], samples[name_i]]))
                X, Y = np.meshgrid(grids_1d[name_j], grids_1d[name_i])
                density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                levels = _hdi_levels(density, hdi_probs)
                ax.contourf(X, Y, density, levels=[*levels, density.max()], cmap="Blues")

            if i == n - 1:
                ax.set_xlabel(name_j, fontsize=textsize)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(name_i, fontsize=textsize)
            elif j == 0:
                ax.set_ylabel("Density", fontsize=textsize)
            if j != 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    return fig


def plot_inversion_result(result: InversionResult) -> Figure:
    """Compare the best, median, and ensemble profiles from one inversion run."""
    return plot_velocity_models(
        {"Best": result.best, "Median": result.median, "Ensemble": result.ensemble},
        show_std=True,
    )
