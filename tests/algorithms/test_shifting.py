import numpy as np
import pytest

from sigpipe.algorithms.shifting.shifting import shift
from sigpipe.base.arrivals import Arrival, TraceArrivals
from sigpipe.base.stream import Stream
from sigpipe.transformers import Shift


def test_a_late_trigger_drops_the_first_samples(stream: Stream) -> None:
    out = shift(stream, t0=0.03)  # 3 samples at 100 Hz
    np.testing.assert_array_equal(out.xt[:, :-3], stream.xt[:, 3:])
    np.testing.assert_array_equal(out.xt[:, -3:], 0.0)
    np.testing.assert_array_equal(out.ts, stream.ts)


def test_an_early_trigger_pads_the_start(stream: Stream) -> None:
    out = shift(stream, t0=-0.02)
    np.testing.assert_array_equal(out.xt[:, :2], 0.0)
    np.testing.assert_array_equal(out.xt[:, 2:], stream.xt[:, :-2])


def test_arrival_times_follow_the_origin(stream: Stream) -> None:
    arrivals = tuple(
        TraceArrivals(arrivals=(Arrival(label="P", time=0.1 * (i + 1), amplitude=1.0),))
        for i in range(stream.nx)
    )
    picked = Stream(
        xt=stream.xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
        arrivals=arrivals,
    )
    out = shift(picked, t0=0.05)
    assert out.arrivals is not None
    times = [trace.arrivals[0].time for trace in out.arrivals]
    np.testing.assert_allclose(times, [0.05, 0.15, 0.25, 0.35])


def test_a_shift_as_long_as_the_record_raises(stream: Stream) -> None:
    with pytest.raises(ValueError, match="shorter than the record"):
        shift(stream, t0=0.5)


def test_no_shift_passes_through(stream: Stream) -> None:
    assert Shift().transform([stream]) == [stream]
