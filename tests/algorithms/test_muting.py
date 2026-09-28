import numpy as np
import pytest

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream
from sigpipe.transformers import Mute


@pytest.fixture
def shot() -> Stream:
    receivers = (Coordinate(10.0, 0.0, 0.0), Coordinate(20.0, 0.0, 0.0))
    return Stream(
        xt=np.ones((2, 1001), dtype=np.float32),
        ts=np.arange(1001, dtype=np.float32) / 1000,
        sampling_freq=1000.0,
        acquisition=LinearAcquisition(source=Coordinate(0.0, 0.0, 0.0), receivers=receivers),
    )


def test_the_surface_wave_window_keeps_the_waves_between_two_velocities(shot: Stream) -> None:
    (kept,) = Mute(method="mute", vmin=100.0, vmax=1000.0, taper=50).transform([shot])

    # At 10 m: from 10 ms (1,000 m/s) to 100 ms (100 m/s), ramps of 50 samples around.
    first = kept.xt[0]
    assert (first[12:100] == 1).all() and (first[150:] == 0).all()
    assert first[125] == pytest.approx(0.5, abs=0.05)  # halfway down the ramp after 100 ms
    # At 20 m: from 20 to 200 ms.
    assert (kept.xt[1][22:200] == 1).all() and (kept.xt[1][250:] == 0).all()


def test_a_ramp_cut_by_the_trace_start_keeps_its_width(shot: Stream) -> None:
    (kept,) = Mute(method="mute", vmax=1000.0, taper=50).transform([shot])

    # 10 and 20 samples of ramp before the window: the ramp's last ones, not a whole ramp
    # squeezed in (which starts at 0).
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(50) / 49))
    assert kept.xt[0][0] == pytest.approx(ramp[40], abs=0.03)
    assert kept.xt[1][0] == pytest.approx(ramp[30], abs=0.03)


def test_a_width_keeps_the_shots_pulse_after_the_slowest_arrival(shot: Stream) -> None:
    (kept,) = Mute(method="mute", vmin=100.0, vmax=1000.0, width=0.05).transform([shot])

    # At 10 m: from 10 ms to 100 ms + 50 ms; at 20 m, to 200 ms + 50 ms.
    assert (kept.xt[0][10:150] == 1).all() and (kept.xt[0][151:] == 0).all()
    assert (kept.xt[1][20:250] == 1).all() and (kept.xt[1][251:] == 0).all()
