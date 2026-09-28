import numpy as np

from sigpipe.algorithms.selection.stream.fk import selection_fk
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream


def test_the_fk_band_takes_each_velocity_given() -> None:
    # A 20 Hz wave moving along the line at 200 m/s, 24 receivers 1 m apart.
    sampling = 500.0
    ts = np.arange(1000) / sampling
    xs = np.arange(24, dtype=float)
    xt = np.sin(2 * np.pi * 20 * (ts[None, :] - xs[:, None] / 200)).astype(np.float32)
    receivers = tuple(Coordinate(float(x), 0.0, 0.0) for x in xs)
    stream = Stream(
        xt=xt,
        ts=ts,
        sampling_freq=sampling,
        acquisition=LinearAcquisition(source=Coordinate(-1.0, 0.0, 0.0), receivers=receivers),
    )

    # No bound: all of its energy, one-sided, kept; above 500 m/s only, none of it.
    assert selection_fk(stream, threshold=0.5) is stream
    assert selection_fk(stream, threshold=0.5, vmin=500.0) is None
