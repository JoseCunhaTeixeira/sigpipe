"""The kept models' marginals: each value's density, and each pair's."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.contour import ContourSet
from scipy.stats import gaussian_kde

from sigpipe.dataio.inversion.plotting import plot_posterior_marginals


def test_two_values_that_move_together_show_their_samples() -> None:
    # One layer spans both depths: the Vs at 1.3 m and at 4.3 m are the same in every model,
    # their pair without a density.
    vs = 230.0 + (np.arange(30) % 9) * 0.5
    with pytest.raises(np.linalg.LinAlgError):
        gaussian_kde(np.vstack([vs, vs]))
    noise = 0.35 + np.random.default_rng(0).normal(0.0, 0.01, 30)
    samples = {"Vs at 1.3 m [m/s]": vs, "Vs at 4.3 m [m/s]": vs.copy(), "Noise factor": noise}

    figure = plot_posterior_marginals(samples)

    # Row by row: the pair (second row, first column) its samples, the others their densities.
    pair, *others = figure.axes[3], figure.axes[6], figure.axes[7]
    assert [type(one) for one in pair.collections] == [PathCollection]
    assert all(not isinstance(one, PathCollection) for ax in others for one in ax.collections)
    plt.close(figure)


def _stretched(n: int, spread: float) -> dict[str, np.ndarray]:
    """The Vs at two depths, near 250 m/s in every model but one, at 800 m/s (a chain's
    excursion): the grid the densities are drawn on spans them all, its cells wider than the
    density of the others."""
    rng = np.random.default_rng(0)
    return {
        name: np.append(rng.normal(250.0, spread, n), 800.0)
        for name in ("Vs at 1.7 m [m/s]", "Vs at 9.2 m [m/s]")
    }


def test_a_density_narrower_than_the_grid_fills_the_bands_it_has() -> None:
    # Half the pair's density on its peak's node: the 30 % threshold is the peak, no band between
    # them, the 60 and 90 % thresholds one node's (contourf refused the four levels as they were:
    # "Contour levels must be increasing", and the figure was lost).
    figure = plot_posterior_marginals(_stretched(2000, 1.0))

    (bands,) = figure.axes[2].collections
    assert isinstance(bands, ContourSet)
    assert len(bands.levels) < 4 and np.all(np.diff(bands.levels) > 0)
    plt.close(figure)


def test_a_density_one_node_holds_shows_its_samples() -> None:
    # 98 % of the pair's density on its peak's node: every threshold is the peak, no band to fill,
    # the samples themselves.
    figure = plot_posterior_marginals(_stretched(4000, 0.2))

    assert [type(one) for one in figure.axes[2].collections] == [PathCollection]
    plt.close(figure)
