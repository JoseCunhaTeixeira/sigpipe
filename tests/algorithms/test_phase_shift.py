"""The phase shift: on a regular grid of offsets, Horner's rule gives the image the sum over the
traces gives, whatever the grid's gaps and shared offsets."""

import numpy as np
import pytest
from scipy.fft import rfft

from sigpipe.algorithms.dispersion.phase_shift import (
    compute_frequency_vector,
    phase_shift,
    regular_grid,
)


def _summed(xt: np.ndarray, sampling_freq: float, offsets: np.ndarray) -> np.ndarray:
    """The image as the sum over the traces makes it, velocity after velocity."""
    xt = xt * np.sqrt(offsets[:, None])
    xf = np.array(rfft(xt, axis=1), dtype=np.complex64)
    fs = compute_frequency_vector(nt=xt.shape[1], sampling_freq=sampling_freq)
    mask = (fs >= 0.0) & (fs <= 100.0)
    fs, xf = fs[mask], xf[:, mask]
    xf_norm = xf / (np.abs(xf) + 1e-12)
    vs = np.linspace(50.0, 1000.0, 200)
    image = np.zeros((fs.size, vs.size), dtype=np.float32)
    for i, v in enumerate(vs):
        dphi = 2 * np.pi * offsets[:, None] * fs / (v + 1e-12)
        image[:, i] = np.abs(np.sum(xf_norm * np.exp(1j * dphi), axis=0))
    return image / len(np.unique(offsets))


@pytest.mark.parametrize(
    "offsets",
    [
        4.5 + 1.5 * np.arange(12),  # a shot off the window's end
        np.array([4.5, 6.0, 9.0, 10.5, 12.0]),  # a receiver left out
        np.array([3.0, 1.5, 0.0, 1.5, 3.0]),  # a shot within the window: offsets shared
        np.array([7.0]),
        np.array([1.0, 2.3, 4.1, 4.2]),  # no regular grid: summed over the traces
    ],
)
def test_the_image_is_the_sum_over_the_traces(offsets: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    xt = rng.standard_normal((offsets.size, 400)).astype(np.float32)

    _, _, image = phase_shift(
        xt, 500.0, offsets, fmin=0.0, fmax=100.0, vmin=50.0, vmax=1000.0, nv=200
    )

    np.testing.assert_allclose(image, _summed(xt, 500.0, offsets), rtol=1e-5, atol=1e-6)


def test_a_grid_holds_every_offset_on_its_steps() -> None:
    found = regular_grid(np.array([4.5, 6.0, 9.0, 4.5]))

    assert found is not None
    steps, x0, dx = found
    assert (x0, dx) == (4.5, 1.5)
    assert steps.tolist() == [0, 1, 3, 0]
    assert regular_grid(np.array([1.0, 2.3, 4.1])) is None
    # A grid of many more steps than offsets is no gain: none.
    assert regular_grid(np.array([0.0, 0.01, 50.0])) is None
