"""The kept models' marginals: each value's density, and each pair's."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
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
