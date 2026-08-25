import numpy as np

from sigpipe.algorithms.flipping.flipping import FlipAxis, flip
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.arrivals import Arrival, TraceArrivals
from sigpipe.base.stream import Stream


def test_flip_space_reverses_receiver_order(stream: Stream) -> None:
    out = flip(stream, axis=FlipAxis.SPACE)
    np.testing.assert_array_equal(out.xt, stream.xt[::-1, :])


def test_flip_space_keeps_acquisition_by_default(stream: Stream) -> None:
    out = flip(stream, axis=FlipAxis.SPACE)
    assert out.acquisition == stream.acquisition


def test_flip_space_reverses_acquisition_when_requested(stream: Stream) -> None:
    out = flip(stream, axis=FlipAxis.SPACE, flip_acquisition=True)
    assert out.acquisition.receivers == tuple(reversed(stream.acquisition.receivers))
    assert out.acquisition.source == stream.acquisition.source


def test_flip_time_reverses_samples_not_receivers(stream: Stream) -> None:
    out = flip(stream, axis=FlipAxis.TIME)
    np.testing.assert_array_equal(out.xt, stream.xt[:, ::-1])
    assert out.acquisition == stream.acquisition


def test_flip_accepts_string_axis(stream: Stream) -> None:
    out = flip(stream, axis="space")
    np.testing.assert_array_equal(out.xt, stream.xt[::-1, :])


def test_flip_reverses_arrivals(linear_acquisition: LinearAcquisition) -> None:
    nt = 10
    xt = np.zeros((len(linear_acquisition.receivers), nt))
    ts = np.arange(nt, dtype=np.float32)
    arrivals = tuple(
        TraceArrivals((Arrival(label="P", time=float(i), amplitude=1.0),))
        for i in range(len(linear_acquisition.receivers))
    )
    s = Stream(
        xt=xt,
        ts=ts,
        sampling_freq=1.0,
        acquisition=linear_acquisition,
        arrivals=arrivals,
    )

    out = flip(s, axis=FlipAxis.SPACE)
    assert out.arrivals == tuple(reversed(arrivals))


def test_flip_preserves_none_arrivals(stream: Stream) -> None:
    assert stream.arrivals is None
    out = flip(stream, axis=FlipAxis.SPACE)
    assert out.arrivals is None
