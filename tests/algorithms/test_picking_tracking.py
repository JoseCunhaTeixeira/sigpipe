import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from sigpipe.algorithms.dispersion.phase_shift import phase_shift
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.algorithms.picking.dispersion.tracking.ridges import (
    corridor,
    followed_ridge,
    lowest_ridge,
    track,
)
from sigpipe.base import Coordinate, DispersionImage, LinearAcquisition, Mode, VelocityType

type Dispersion = Callable[[np.ndarray], np.ndarray]

# A synthetic line longer than the demo windows, so that its images resolve the modes: 48
# receivers 1 m apart (a 47 m window), 2 s records at 500 Hz, and the default dispersion grid.
SPACING = 1.0  # m
ACQUISITION = LinearAcquisition(
    source=Coordinate(0.0, 0.0, 0.0),
    receivers=tuple(Coordinate(2.0 + k * SPACING, 0.0, 0.0) for k in range(48)),
)
WINDOW_LENGTH = 47.0  # m
SAMPLING_FREQ = 500.0  # Hz
N_SAMPLES = 1_000


def m0(frequencies: np.ndarray) -> np.ndarray:
    """A fundamental mode: 400 m/s at 0 Hz, down to 150 m/s at high frequencies."""
    return 150 + 250 * np.exp(-frequencies / 15)


def m1(frequencies: np.ndarray) -> np.ndarray:
    return 1.8 * m0(frequencies)


def _shot(
    modes: Sequence[tuple[Dispersion, float | Dispersion]], noise: float, seed: int = 0
) -> DispersionImage:
    """The dispersion image of a synthetic shot: each mode a plane wave with phase velocity c(f)
    and amplitude a (or a(f)), plus white noise, through sigpipe's phase shift."""
    rng = np.random.default_rng(seed)
    offsets = ACQUISITION.offsets.astype(float)
    frequencies = np.fft.rfftfreq(N_SAMPLES, 1 / SAMPLING_FREQ)
    shape = (offsets.size, frequencies.size)
    spectra = noise * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)
    for velocity, amplitude in modes:
        scale = amplitude(frequencies) if callable(amplitude) else amplitude
        spectra += scale * np.exp(
            -2j * np.pi * offsets[:, None] * frequencies / velocity(frequencies)
        )
    traces = np.fft.irfft(spectra, n=N_SAMPLES, axis=1)
    fs, vs, fv_map = phase_shift(
        traces, SAMPLING_FREQ, offsets, fmin=0.0, fmax=100.0, vmin=1.0, vmax=1_000.0
    )
    return DispersionImage(
        fv_map=fv_map, fs=fs, vs=vs, type=VelocityType.PHASE, acquisition=ACQUISITION
    )


# ---------------------------------------------------------------- synthetic shots


def test_m0_follows_the_true_curve() -> None:
    (mode,) = pick_modes(_shot([(m0, 1.0)], noise=0.3))
    band = (mode.frequencies >= 5) & (mode.frequencies <= 60)

    assert mode.label == "M0"
    assert mode.kept[band].all()
    # Within 3 %, though at 5 Hz the ridge is several hundred m/s wide.
    assert mode.velocities[band] == pytest.approx(m0(mode.frequencies[band]), rel=0.03)


def test_m0_is_the_lowest_ridge_not_the_brightest() -> None:
    image = _shot([(m0, 0.8), (m1, 1.0)], noise=0.3)
    # At 30 Hz, the brightest value of the image is on M1.
    row = int(np.searchsorted(image.fs, 30.0))
    brightest = float(image.vs[np.argmax(image.fv_map[row])])
    assert brightest == pytest.approx(float(m1(np.array(30.0))), rel=0.05)

    modes = pick_modes(image, PickingParameters(max_modes=2))

    assert [mode.label for mode in modes] == ["M0", "M1"]
    for mode, own, other in [(modes[0], m0, m1), (modes[1], m1, m0)]:
        band = mode.kept & (mode.frequencies >= 20) & (mode.frequencies <= 60)
        f, v = mode.frequencies[band], mode.velocities[band]
        # Each mode stays on its own branch.
        assert np.mean(np.abs(v - own(f)) < np.abs(v - other(f))) > 0.9


def test_the_default_is_m0_only() -> None:
    modes = pick_modes(_shot([(m0, 0.8), (m1, 1.0)], noise=0.3))

    assert [mode.label for mode in modes] == ["M0"]


def test_the_pick_stops_where_its_ridge_breaks() -> None:
    # At 40 Hz the ridge jumps onto a branch 60 % faster: two runs, and the wider one is kept.
    def jumping(frequencies: np.ndarray) -> np.ndarray:
        return np.where(frequencies < 40, m0(frequencies), 1.6 * m0(frequencies))

    image = _shot([(jumping, 1.0)], noise=0.3)

    (mode,) = pick_modes(image)
    (everything,) = pick_modes(image, PickingParameters(max_gap_hz=None))

    assert mode.frequencies[mode.kept].min() >= 40
    assert everything.frequencies[everything.kept].min() < 40


def test_the_pick_goes_on_over_a_dimmer_ridge_under_it() -> None:
    # Below 20 Hz, a slow wave under M0, dimmer but over 0.35 of its column's maximum: the lowest
    # ridge there. Past its run the pick follows M0 on, as over a short window's sidelobes and
    # aliases at its band's ends (on active_p2).
    def slow(frequencies: np.ndarray) -> np.ndarray:
        return np.full_like(frequencies, 60.0)

    def below_20_hz(frequencies: np.ndarray) -> np.ndarray:
        return np.where(frequencies < 20, 0.6, 0.0)

    (mode,) = pick_modes(_shot([(m0, 1.0), (slow, below_20_hz)], noise=0.3))
    band = (mode.frequencies >= 6) & (mode.frequencies < 20)

    assert mode.kept[band].all()
    assert mode.velocities[band] == pytest.approx(m0(mode.frequencies[band]), rel=0.05)


def test_a_brighter_ridge_under_the_one_followed_ends_the_following() -> None:
    # Above 40 Hz a branch 60 % faster than M0 alone, the pick (the widest run); below, M0
    # appears under it, brighter than the branch going on: the fundamental mode, not a sidelobe.
    def fast(frequencies: np.ndarray) -> np.ndarray:
        return 1.6 * m0(frequencies)

    def dimmer_below_40_hz(frequencies: np.ndarray) -> np.ndarray:
        return np.where(frequencies < 40, 0.8, 1.0)

    def below_40_hz(frequencies: np.ndarray) -> np.ndarray:
        return np.where(frequencies < 40, 1.0, 0.0)

    (mode,) = pick_modes(_shot([(fast, dimmer_below_40_hz), (m0, below_40_hz)], noise=0.3))
    below = (mode.frequencies >= 6) & (mode.frequencies <= 35)

    # The pick stays the branch's run; below it, M0 is tracked, not the dimmer branch.
    assert mode.frequencies[mode.kept].min() >= 40
    f, v = mode.frequencies[below], mode.velocities[below]
    assert np.mean(np.abs(v - m0(f)) < np.abs(v - fast(f))) > 0.9


def test_a_band_cut_keeps_the_pick_within() -> None:
    # G3's first fix for a mode jump: the frequencies searched stop where the jump starts.
    (mode,) = pick_modes(_shot([(m0, 1.0)], noise=0.3), PickingParameters(fmin=12.0, fmax=30.0))

    kept = mode.frequencies[mode.kept]
    assert kept.min() >= 12.0 and kept.max() <= 30.0
    assert kept.max() - kept.min() > 10


def _silent(low: float, high: float) -> Dispersion:
    """An amplitude of 1 except between `low` and `high` Hz, where the wave carries nothing."""
    return lambda frequencies: np.where((frequencies >= low) & (frequencies < high), 0.0, 1.0)


def test_a_short_gap_in_the_ridge_is_bridged_a_wide_one_ends_it() -> None:
    (short,) = pick_modes(_shot([(m0, _silent(30.0, 30.5))], noise=0.3))
    (wide,) = pick_modes(_shot([(m0, _silent(30.0, 36.0))], noise=0.3))

    # One silent column drops 1.5 Hz of them, under 2 Hz: one run on both sides.
    kept = short.frequencies[short.kept]
    assert kept.min() < 20 and kept.max() > 60
    # 6 Hz: the run ends there, and the wider side (36 to 100 Hz) is the pick.
    kept = wide.frequencies[wide.kept]
    assert kept.min() >= 36


@pytest.mark.parametrize("spacings", [1.0, 2.0])
def test_the_search_stays_between_its_shortest_and_longest_wavelengths(spacings: float) -> None:
    parameters = PickingParameters(max_wavelength=1.0, min_wavelength=spacings)
    (mode,) = pick_modes(_shot([(m0, 1.0)], noise=0.3), parameters)
    f, v = mode.frequencies, mode.velocities

    # Wavelengths from `spacings` receiver spacings (one by default, two the spatial Nyquist
    # limit) up to the window length.
    assert (v >= spacings * SPACING * f).all()
    assert (v <= WINDOW_LENGTH * f).all()
    # Points pinned to a bound, or below the noise floor, are never kept.
    assert mode.pinned.any()
    assert not (mode.kept & mode.pinned).any()
    assert (mode.coherence[mode.kept] >= mode.noise_floor).all()


RIDGE_FREQUENCIES = np.arange(5.0, 60.5, 0.5)  # Hz


def _ridge_image(heights: float | np.ndarray) -> DispersionImage:
    """A noise-free ridge along M0, `heights` times the noise floor at its peak: one height for
    all frequencies, or one per frequency."""
    vs = np.arange(1.0, 1_001.0)
    velocities = m0(RIDGE_FREQUENCIES)[:, None]
    ridge = np.exp(-0.5 * ((vs - velocities) / (0.1 * velocities)) ** 2)
    peaks = np.broadcast_to(heights, RIDGE_FREQUENCIES.shape)[:, None]
    noise_floor = 1 / math.sqrt(len(ACQUISITION.receivers))
    return DispersionImage(
        fv_map=peaks * noise_floor * ridge,
        fs=RIDGE_FREQUENCIES,
        vs=vs,
        type=VelocityType.PHASE,
        acquisition=ACQUISITION,
    )


@pytest.mark.parametrize(
    ("height", "labels"),
    [
        pytest.param(0.5, [], id="peak-below-the-floor"),
        pytest.param(1.2, [], id="peak-under-1.5-floors"),
        pytest.param(2.0, ["M0"], id="peak-at-2-floors"),
    ],
)
def test_a_mode_needs_a_ridge_above_the_noise_floor(height: float, labels: list[str]) -> None:
    assert [mode.label for mode in pick_modes(_ridge_image(height))] == labels


def test_points_below_the_noise_floor_are_dropped() -> None:
    # A faint mode: 1.6 times the noise floor, but 0.9 times from 30 to 40 Hz. Those points are
    # above half the mode's median coherence, so only the noise floor drops them.
    faint = (RIDGE_FREQUENCIES >= 30) & (RIDGE_FREQUENCIES <= 40)

    # The ridge's continuity aside: 10 Hz of dropped points would end the pick's run.
    parameters = PickingParameters(max_gap_hz=None)
    (mode,) = pick_modes(_ridge_image(np.where(faint, 0.9, 1.6)), parameters)

    assert not mode.kept[faint].any()
    assert mode.kept[~faint].all()


def test_a_ridge_cut_by_a_bound_is_not_kept() -> None:
    # The relative-coherence rule nearly off, so only the pinning can drop these points.
    parameters = PickingParameters(max_wavelength=1.0, min_relative_coherence=0.01)
    (mode,) = pick_modes(_shot([(m0, 1.0)], noise=0.3), parameters)

    # Below 5 Hz, M0's wavelength is over 1.4 times the window length: its ridge lies above the
    # search, so the bound decides instead of the data, even where the smoothing moved the pick
    # off the corridor's edge.
    assert not mode.kept[mode.frequencies < 5].any()


def test_nothing_is_kept_where_m0_is_below_the_search_floor() -> None:
    # Stopped at the spatial Nyquist limit (two spacings): above 76 Hz, M0 is slower than it
    # allows. Above that floor, the first sidelobe of its ridge still beats the noise floor, but
    # not half the mode's coherence.
    (mode,) = pick_modes(_shot([(m0, 1.0)], noise=0.3), PickingParameters(min_wavelength=2.0))
    f = mode.frequencies

    assert not mode.kept[m0(f) < 2 * SPACING * f].any()


def test_a_clear_ridge_is_followed_into_the_aliasing_zone_by_default() -> None:
    # Down to one spacing: M0 kept on to 100 Hz, its points under two spacings among them, for
    # the checks to flag.
    (mode,) = pick_modes(_shot([(m0, 1.0)], noise=0.3))
    f, kept = mode.frequencies, mode.kept

    assert f[kept].max() == 100.0
    assert kept[m0(f) < 2 * SPACING * f].all()


@pytest.mark.parametrize("zero_hz_row", ["flat", "ridge"])
def test_0_hz_is_never_kept(zero_hz_row: str) -> None:
    # A ridge at 55 m/s, above a 50 m/s vmin. In a phase-shift image, the 0 Hz row is flat, as no
    # velocity shifts a phase at 0 Hz; another transform could show the ridge there too.
    fs = np.arange(0.0, 10.5, 0.5)
    vs = np.arange(50.0, 101.0)
    fv_map = np.tile(0.05 + 0.85 * np.exp(-0.5 * ((vs - 55) / 3) ** 2), (fs.size, 1))
    if zero_hz_row == "flat":
        fv_map[0] = 0.9
    image = DispersionImage(
        fv_map=fv_map, fs=fs, vs=vs, type=VelocityType.PHASE, acquisition=ACQUISITION
    )

    (mode,) = pick_modes(image)

    assert not mode.kept[mode.frequencies == 0].any()


def test_the_curve_is_the_kept_points_over_wavelength() -> None:
    (mode,) = pick_modes(_shot([(m0, 1.0)], noise=0.3))
    curve = mode.curve

    assert curve is not None
    assert curve.mode == Mode("M", 0)
    assert curve.vs_err is not None
    assert (curve.vs_err > 0).all()
    # Like PAC's picks: one point per metre of wavelength, sorted by frequency.
    assert (np.diff(curve.fs) > 0).all()
    assert np.diff(curve.vs / curve.fs) == pytest.approx(-1.0, rel=1e-3)  # float32
    kept_wavelengths = mode.velocities[mode.kept] / mode.frequencies[mode.kept]
    assert kept_wavelengths.min() <= (curve.vs / curve.fs).min()
    assert (curve.vs / curve.fs).max() <= kept_wavelengths.max()


# ---------------------------------------------------------------- the three stages


@pytest.mark.parametrize(
    ("column", "start", "stop", "expected"),
    [
        # A bump under 0.35 times the brightest value is not a ridge.
        pytest.param([0.1, 0.3, 0.2, 0.6, 0.3, 1.0, 0.4], 0, 6, 3, id="first-bump-over-threshold"),
        pytest.param([0.9, 0.2, 0.1, 0.5, 0.2, 0.1], 1, 5, 3, id="searched-from-start"),
        pytest.param([0.1, 0.2, 0.4, 0.8, 1.0, 0.3], 0, 4, 4, id="still-rising-at-stop"),
        # Even past a bump: the ridge above stop can throw sidelobes below it.
        pytest.param([0.1, 0.5, 0.3, 0.6, 0.8, 1.0], 0, 5, 5, id="rising-at-stop-past-a-bump"),
        pytest.param([0.3, 1.0, 0.8, 0.5, 0.3, 0.2], 1, 5, 1, id="still-falling-at-start"),
    ],
)
def test_lowest_ridge(column: list[float], start: int, stop: int, expected: int) -> None:
    ridge = lowest_ridge(np.array([column]), np.array([start]), np.array([stop]), threshold=0.35)

    assert ridge.tolist() == [expected]


@pytest.mark.parametrize(
    ("ridge", "start", "stop", "half_width", "expected"),
    [
        pytest.param(50, 0, 100, 0.2, (20, 80), id="ridge-plus-or-minus-20-percent"),
        pytest.param(50, 30, 70, 0.2, (30, 70), id="within-start-and-stop"),
        pytest.param(50, 0, 100, 0.001, (50, 51), id="at-least-two-cells"),
        pytest.param(100, 0, 100, 0.001, (99, 100), id="two-cells-at-stop"),
    ],
)
def test_corridor(
    ridge: int, start: int, stop: int, half_width: float, expected: tuple[int, int]
) -> None:
    velocities = np.arange(100.0, 201.0)  # index i is 100 + i m/s

    low, high = corridor(
        velocities, np.array([ridge]), np.array([start]), np.array([stop]), half_width
    )

    assert (int(low[0]), int(high[0])) == expected


# Three frequencies, 1 Hz apart; velocities from 100 to 200 m/s, 10 m/s apart.
TRACK_VELOCITIES = np.linspace(100.0, 200.0, 11)
TRACK_FREQUENCIES = np.array([10.0, 11.0, 12.0])


def test_track_without_smoothness_takes_the_brightest_cells() -> None:
    image = np.zeros((3, 11))
    image[0, 5] = image[1, 2] = image[2, 8] = 1.0
    low, high = np.zeros(3, dtype=int), np.full(3, 10)

    path, on_edge = track(image, TRACK_VELOCITIES, TRACK_FREQUENCIES, low, high, smoothness=0.0)

    assert path.tolist() == [5, 2, 8]
    assert not on_edge.any()


def test_smoothness_keeps_the_track_off_a_lone_bright_cell() -> None:
    image = np.zeros((3, 11))
    image[:, 5] = 0.6
    image[1, 9] = 1.0
    low, high = np.zeros(3, dtype=int), np.full(3, 10)

    free, _ = track(image, TRACK_VELOCITIES, TRACK_FREQUENCIES, low, high, smoothness=0.0)
    smooth, _ = track(image, TRACK_VELOCITIES, TRACK_FREQUENCIES, low, high, smoothness=10.0)

    assert free.tolist() == [5, 9, 5]
    assert smooth.tolist() == [5, 5, 5]


def test_track_stays_in_its_corridor_and_flags_its_edges() -> None:
    image = np.zeros((3, 11))
    image[0, 3] = image[1, 5] = image[2, 7] = 1.0
    image[1, 9] = 2.0  # brighter, but outside the corridor
    low, high = np.full(3, 3), np.full(3, 7)

    path, on_edge = track(image, TRACK_VELOCITIES, TRACK_FREQUENCIES, low, high, smoothness=0.0)

    assert path.tolist() == [3, 5, 7]
    assert on_edge.tolist() == [True, False, True]


@pytest.mark.parametrize(
    ("height", "expected"),
    [
        pytest.param(0.5, [6, 6, 6, 6, 6], id="dimmer-bump-under-it"),
        pytest.param(1.5, [6, 6, 6, 2, 2], id="brighter-bump-under-it"),
    ],
)
def test_followed_ridge(height: float, expected: list[int]) -> None:
    # A pick on the first three of five frequencies, its ridge at 160 m/s; on the last two, a
    # bump at 120 m/s, the lowest ridge there, 25 % under it: further than the 20 % a ridge is
    # followed across.
    image = np.full((5, 11), 0.05)
    image[:, 6] = 1.0
    image[3:, 2] = height
    start, stop = np.zeros(5, dtype=int), np.full(5, 10)

    ridge = followed_ridge(image, TRACK_VELOCITIES, start, stop, 0.35, jump=0.2, stretch=(0, 2))

    assert ridge.tolist() == expected


def test_a_guide_centres_the_corridor_on_the_curve_given() -> None:
    """Two ridges of the same height; the lowest is M0 by default, the guide picks the one it
    follows (G4's re-pick on the neighbouring windows' median curve)."""
    vs = np.arange(1.0, 1_001.0)
    noise_floor = 1 / math.sqrt(len(ACQUISITION.receivers))

    def ridge(velocities: np.ndarray) -> np.ndarray:
        centres = velocities[:, None]
        return 2.0 * noise_floor * np.exp(-0.5 * ((vs - centres) / (0.1 * centres)) ** 2)

    lower, upper = m0(RIDGE_FREQUENCIES), 1.8 * m0(RIDGE_FREQUENCIES)
    image = DispersionImage(
        fv_map=np.maximum(ridge(lower), ridge(upper)),
        fs=RIDGE_FREQUENCIES,
        vs=vs,
        type=VelocityType.PHASE,
        acquisition=ACQUISITION,
    )
    guide = tuple(
        (float(f), float(v)) for f, v in zip(RIDGE_FREQUENCIES[::5], upper[::5], strict=True)
    )

    (default,) = pick_modes(image, PickingParameters())
    (guided,) = pick_modes(image, PickingParameters(guide=guide))

    assert default.velocities[default.kept] == pytest.approx(lower[default.kept], rel=0.05)
    assert guided.velocities[guided.kept] == pytest.approx(upper[guided.kept], rel=0.05)
