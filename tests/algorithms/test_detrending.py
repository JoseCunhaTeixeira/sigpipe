import numpy as np

from sigpipe.algorithms.detrending.constant import detrend_constant
from sigpipe.algorithms.detrending.linear import detrend_linear
from sigpipe.algorithms.detrending.registry import DETRENDING_METHODS
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.stream import Stream


def _stream_with_trend(acquisition: LinearAcquisition, offset: float, slope: float) -> Stream:
    nt = 50
    ts = np.arange(nt, dtype=np.float32)
    trace = offset + slope * ts
    xt = np.tile(trace, (len(acquisition.receivers), 1))
    return Stream(xt=xt, ts=ts, sampling_freq=1.0, acquisition=acquisition)


def test_registry_exposes_expected_methods() -> None:
    assert {
        "linear": detrend_linear,
        "constant": detrend_constant,
    } == DETRENDING_METHODS


def test_detrend_constant_removes_mean(linear_acquisition: LinearAcquisition) -> None:
    s = _stream_with_trend(linear_acquisition, offset=10.0, slope=0.0)
    out = detrend_constant(s)
    np.testing.assert_allclose(out.xt.mean(axis=1), 0.0, atol=1e-4)


def test_detrend_constant_preserves_metadata(linear_acquisition: LinearAcquisition) -> None:
    s = _stream_with_trend(linear_acquisition, offset=10.0, slope=0.0)
    out = detrend_constant(s)
    np.testing.assert_array_equal(out.ts, s.ts)
    assert out.sampling_freq == s.sampling_freq
    assert out.acquisition == s.acquisition


def test_detrend_linear_removes_linear_trend(linear_acquisition: LinearAcquisition) -> None:
    s = _stream_with_trend(linear_acquisition, offset=5.0, slope=0.3)
    out = detrend_linear(s)
    # A pure linear trend is fully removed, leaving ~0 everywhere.
    np.testing.assert_allclose(out.xt, 0.0, atol=1e-3)
