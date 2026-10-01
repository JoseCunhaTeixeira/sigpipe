"""The surface waves' cone on synthetic shots of known velocity, and the energy arriving faster
than it: a refraction planted, measured; none, none measured."""

import numpy as np

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream
from sigpipe.masw.quality.cone import MIN_TRACES, fast_share, peak_velocities, surface_wave_cone

SAMPLING = 1000.0
N_TRACES = 24
DX = 1.0
VELOCITY = 200.0  # m/s, the surface wave
FREQUENCY = 20.0  # Hz, its wavelet's centre
REFRACTION = 1200.0  # m/s, a head wave arriving before it


def _berlage(tau: np.ndarray) -> np.ndarray:
    """A causal wavelet starting at tau = 0, peaking at 1."""
    pulse = np.where(
        tau >= 0, tau * np.exp(-2 * FREQUENCY * tau) * np.cos(2 * np.pi * FREQUENCY * tau), 0.0
    )
    return pulse / (np.exp(-1) / (2 * FREQUENCY))


def _shot(refraction: float = 0.0, shot_s: float = 0.0) -> Stream:
    """A surface wave moving out at VELOCITY from a source 2 m before the first receiver, the
    shot at `shot_s`, over weak noise; with `refraction`, a head wave of that amplitude relative
    to the surface wave, at REFRACTION."""
    rng = np.random.default_rng(0)
    ts = np.arange(int(1.0 * SAMPLING)) / SAMPLING
    receivers = tuple(Coordinate(2.0 + i * DX, 0.0, 0.0) for i in range(N_TRACES))
    acquisition = LinearAcquisition(source=Coordinate(0.0, 0.0, 0.0), receivers=receivers)
    xt = rng.standard_normal((N_TRACES, ts.size)) * 0.01
    for i, offset in enumerate(acquisition.offsets):
        decay = 1 / np.sqrt(offset / 2.0)
        xt[i] += _berlage(ts - shot_s - offset / VELOCITY) * decay
        xt[i] += refraction * _berlage(ts - shot_s - offset / REFRACTION) * decay
    return Stream(
        xt=xt.astype(np.float32),
        ts=ts.astype(np.float32),
        sampling_freq=SAMPLING,
        acquisition=acquisition,
    )


def test_the_cone_brackets_the_surface_waves_from_their_peaks() -> None:
    velocities = peak_velocities(_shot(shot_s=0.1), shot_s=0.1, min_offset_m=3 * DX)

    cone = surface_wave_cone([velocities])

    # Traces nearer than 3 m left out; each peak a little after its arrival (the wavelet's
    # delay), so the apparent velocities sit below the wave's, nearer it far from the shot.
    assert velocities.size == N_TRACES - 1
    assert cone is not None and cone.n_traces == N_TRACES - 1
    assert 100 < cone.vmin < cone.median < cone.vmax < VELOCITY
    # Widened as the trial does, the cone keeps the wave's onset.
    assert cone.vmax * 1.5 > VELOCITY


def test_too_few_traces_draw_no_cone() -> None:
    assert surface_wave_cone([np.full(MIN_TRACES - 1, 200.0)]) is None
    assert surface_wave_cone([]) is None


def test_energy_faster_than_the_cone_is_measured() -> None:
    measured = {
        refraction: fast_share(
            _shot(refraction), 0.0, vmax=1.5 * VELOCITY, slowest=80.0, pad_s=0.05, min_offset_m=3
        )
        for refraction in (0.0, 1.0)
    }

    # A clean shot: next to nothing before the cone; a head wave as strong as the surface
    # wave: several times more (its wavelet's tail crosses the cone's edge near the shot).
    assert measured[0.0] is not None and measured[0.0] < 0.05
    assert measured[1.0] is not None and measured[1.0] > 0.15
