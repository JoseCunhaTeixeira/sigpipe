import numpy as np
import pytest

from sigpipe.algorithms.padding.padding import pad
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.stream import Stream


def test_pad_appends_zeros(stream: Stream) -> None:
    out = pad(stream, n=10)
    assert out.nt == stream.nt + 10
    np.testing.assert_array_equal(out.xt[:, : stream.nt], stream.xt)
    np.testing.assert_allclose(out.xt[:, stream.nt :], 0.0)


def test_pad_extends_ts_at_same_sampling_rate(stream: Stream) -> None:
    out = pad(stream, n=10)
    dt = 1.0 / stream.sampling_freq
    np.testing.assert_allclose(np.diff(out.ts), dt, atol=1e-5)
    assert out.ts[0] == stream.ts[0]


def test_pad_zero_n_is_a_noop_on_length(stream: Stream) -> None:
    out = pad(stream, n=0)
    assert out.nt == stream.nt
    np.testing.assert_array_equal(out.xt, stream.xt)


def test_pad_negative_n_raises(stream: Stream) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        pad(stream, n=-1)


def test_pad_taper_exceeding_nt_raises(stream: Stream) -> None:
    with pytest.raises(ValueError, match="cannot exceed nt"):
        pad(stream, n=5, taper=stream.nt + 1)


def test_pad_taper_tapers_trailing_samples(linear_acquisition: LinearAcquisition) -> None:
    nt = 20
    xt = np.ones((len(linear_acquisition.receivers), nt))
    ts = np.arange(nt, dtype=np.float32)
    s = Stream(xt=xt, ts=ts, sampling_freq=1.0, acquisition=linear_acquisition)

    out = pad(s, n=0, taper=10)

    # First sample of the taper window is ~unattenuated, last is ~fully attenuated.
    assert out.xt[0, -10] == pytest.approx(1.0, abs=1e-3)
    assert out.xt[0, -1] == pytest.approx(0.0, abs=1e-3)
