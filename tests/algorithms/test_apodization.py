import numpy as np

from sigpipe.algorithms.apodization.hanning import apodize_hanning
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.stream import Stream


def _constant_stream(acquisition: LinearAcquisition, nt: int = 40) -> Stream:
    xt = np.ones((len(acquisition.receivers), nt))
    ts = np.arange(nt, dtype=np.float32)
    return Stream(xt=xt, ts=ts, sampling_freq=1.0, acquisition=acquisition)


def test_apodize_tapers_edges_to_zero(linear_acquisition: LinearAcquisition) -> None:
    s = _constant_stream(linear_acquisition)
    out = apodize_hanning(s, frac=0.1)
    np.testing.assert_allclose(out.xt[:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(out.xt[:, -1], 0.0, atol=1e-6)


def test_apodize_leaves_center_untouched(linear_acquisition: LinearAcquisition) -> None:
    s = _constant_stream(linear_acquisition)
    out = apodize_hanning(s, frac=0.1)
    n_taper = max(1, int(0.1 * s.nt))
    center = out.xt[:, n_taper:-n_taper]
    np.testing.assert_allclose(center, 1.0)


def test_apodize_preserves_shape_and_metadata(stream: Stream) -> None:
    out = apodize_hanning(stream)
    assert out.xt.shape == stream.xt.shape
    np.testing.assert_array_equal(out.ts, stream.ts)
    assert out.acquisition == stream.acquisition
