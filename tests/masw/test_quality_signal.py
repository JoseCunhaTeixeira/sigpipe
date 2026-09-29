"""The signal measures on synthetic records with a known answer: a surface wave of known
velocity and known noise, with defects planted one at a time."""

from dataclasses import replace

import numpy as np
import pytest

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream
from sigpipe.masw.quality.measures import SignalLimits, measure_signal
from sigpipe.masw.quality.signal import (
    Windows,
    dead_clipped_nan,
    first_breaks,
    lateral_coherence,
    pulse_durations,
    rms_decay_outliers,
    signal_windows,
    snr_db,
    snr_reach,
    spectral_deviations,
    trace_snrs,
    trigger_shift,
    usable_band,
)
from sigpipe.transformers import ActiveShotCorrelation, Apodize, Mute, Stack

SAMPLING = 1000.0
N_TRACES = 24
DX = 1.0
VELOCITY = 200.0  # m/s, the surface wave
FREQUENCY = 20.0  # Hz, the wavelet's centre
# The surface-wave window of PACo's signal QC: from 1,500 to 80 m/s, 50 ms on both sides.
VMIN, VMAX, PAD = 80.0, 1500.0, 0.05


def _shot(noise: float = 0.02, t0: float = 0.0, duration_s: float = 2.0, seed: int = 0) -> Stream:
    """A causal wavelet (Berlage: a damped, delayed cosine, starting at the arrival) moving out
    at VELOCITY from a source 2 m before the first receiver, decaying with offset, over white
    noise of RMS `noise`; it peaks at 1 on the nearest trace."""
    rng = np.random.default_rng(seed)
    ts = np.arange(int(duration_s * SAMPLING)) / SAMPLING
    receivers = tuple(Coordinate(2.0 + i * DX, 0.0, 0.0) for i in range(N_TRACES))
    acquisition = LinearAcquisition(source=Coordinate(0.0, 0.0, 0.0), receivers=receivers)
    xt = rng.standard_normal((N_TRACES, ts.size)) * noise
    for i, offset in enumerate(acquisition.offsets):
        tau = ts - (t0 + offset / VELOCITY)
        pulse = np.where(
            tau >= 0, tau * np.exp(-2 * FREQUENCY * tau) * np.cos(2 * np.pi * FREQUENCY * tau), 0.0
        )
        xt[i] += pulse / (np.exp(-1) / (2 * FREQUENCY)) / np.sqrt(offset / 2.0)
    return Stream(
        xt=xt.astype(np.float32),
        ts=ts.astype(np.float32),
        sampling_freq=SAMPLING,
        acquisition=acquisition,
    )


def _with_trace(stream: Stream, index: int, trace: np.ndarray) -> Stream:
    xt = stream.xt.copy()
    xt[index] = trace
    return replace(stream, xt=xt)


def _windows(stream: Stream) -> Windows:
    windows = signal_windows(
        np.asarray(stream.acquisition.offsets), np.asarray(stream.ts), VMIN, VMAX, PAD
    )
    assert windows is not None
    return windows


def test_the_windows_follow_the_moveout_and_the_noise_comes_after_the_slowest_arrival() -> None:
    stream = _shot()
    windows = _windows(stream)

    assert windows.where.startswith("after the slowest arrival")
    # The far trace's window ends later than the near one's, and both hold their arrival.
    near, far = windows.signal[0], windows.signal[-1]
    ts = np.asarray(stream.ts)
    assert ts[near][-1] < ts[far][-1]
    for i in (0, -1):
        arrival = stream.acquisition.offsets[i] / VELOCITY
        assert windows.signal[i][int(arrival * SAMPLING)]
    assert not (windows.signal & windows.noise).any()
    # A record that ends before the slowest arrival has no noise window.
    offsets = np.asarray(stream.acquisition.offsets)
    assert signal_windows(offsets, np.asarray(stream.ts[:200]), VMIN, VMAX, PAD) is None


def test_dead_clipped_and_nan_traces_are_found() -> None:
    stream = _shot()
    stream = _with_trace(stream, 3, np.zeros(stream.xt.shape[1], dtype=np.float32))
    stream = _with_trace(stream, 7, np.clip(stream.xt[7], -0.02, 0.02))
    nan_trace = stream.xt[11].copy()
    nan_trace[100] = np.nan
    stream = _with_trace(stream, 11, nan_trace)

    dead, clipped, nan = dead_clipped_nan(stream.xt, dead_ratio=0.01, clip_share=0.005)

    assert list(np.flatnonzero(dead)) == [3]
    assert list(np.flatnonzero(clipped)) == [7]
    assert list(np.flatnonzero(nan)) == [11]


def test_an_amplitude_off_the_decay_is_an_outlier() -> None:
    stream = _with_trace(_shot(), 10, _shot().xt[10] * 30)
    rms = np.sqrt(np.mean(stream.xt.astype(float) ** 2, axis=1))

    outliers = rms_decay_outliers(
        rms, np.asarray(stream.acquisition.offsets), np.ones(N_TRACES, bool), 3.0
    )

    assert list(np.flatnonzero(outliers)) == [10]


def test_the_snr_matches_the_noise_planted() -> None:
    quiet, loud = _shot(noise=0.001), _shot(noise=0.05)
    windows = _windows(quiet)

    quiet_snr = np.median(snr_db(quiet.xt.astype(float), windows))
    loud_snr = np.median(snr_db(loud.xt.astype(float), windows))

    # 50 x less noise RMS is 34 dB more, within the window's own share of the wavelet.
    assert 25 < quiet_snr - loud_snr < 40


def test_the_usable_band_holds_the_wavelet_and_no_more() -> None:
    stream = _shot(noise=0.05)

    band = usable_band(stream.xt.astype(float), SAMPLING, _windows(stream), band_db=6.0)

    assert band is not None
    fmin, fmax = band
    # A 20 Hz Berlage pulse has its energy between a few Hz and about 60 Hz.
    assert 0 <= fmin <= 12  # a causal pulse has energy down to 0 Hz
    assert 35 <= fmax <= 80


def test_neighbours_correlate_and_a_reversed_trace_shows_in_the_polarity() -> None:
    stream = _shot()
    windows = _windows(stream)
    max_lag_s = DX / VMIN

    coherence, polarity = lateral_coherence(stream.xt.astype(float), windows, SAMPLING, max_lag_s)
    reversed_stream = _with_trace(stream, 5, -stream.xt[5])
    reversed_coherence, reversed_polarity = lateral_coherence(
        reversed_stream.xt.astype(float), windows, SAMPLING, max_lag_s
    )

    assert coherence.min() > 0.9
    assert (polarity > 0).all()
    assert list(reversed_polarity[4:6]) == [-1, -1] and reversed_coherence.min() > 0.9


def test_the_first_breaks_give_the_trigger_and_the_velocity() -> None:
    for t0 in (0.0, 0.05):
        stream = _shot(t0=t0)
        breaks = first_breaks(
            stream.xt.astype(float), np.asarray(stream.ts), _windows(stream), ratio=5.0
        )
        fit = trigger_shift(breaks, np.asarray(stream.acquisition.offsets))

        assert fit is not None
        shift, velocity, scatter = fit
        assert shift == pytest.approx(t0, abs=0.01)
        assert 150 < velocity < 250
        assert scatter < 0.005


def test_the_pulse_lasts_from_the_first_break_to_its_energys_fall() -> None:
    # The wavelet's envelope, t exp(-2 f t), peaks at 25 ms and falls to a tenth of its peak at
    # about 120 ms: every trace's pulse, whatever its amplitude. A trace without a break has none.
    stream = _shot()
    xt, ts = stream.xt.astype(float), np.asarray(stream.ts, dtype=float)
    breaks = first_breaks(xt, ts, _windows(stream), ratio=5.0)
    breaks[3] = np.nan

    pulses = pulse_durations(xt, ts, breaks, ratio=0.1, longest_s=0.5)

    assert np.isnan(pulses[3])
    assert np.nanmedian(pulses) == pytest.approx(0.12, abs=0.015)
    # Cut short, the search finds no fall.
    assert np.isnan(pulse_durations(xt, ts, breaks, ratio=0.1, longest_s=0.05)).all()


def test_the_reach_is_where_the_traces_median_snr_falls_under_the_limit() -> None:
    # 60 dB at the shot, 0.7 dB less every metre: 6 dB at 77.1 m; bins of 10 m find 77.6.
    offsets = np.arange(0.0, 100.0, 1.0)
    measured = [(offsets, 60 - 0.7 * offsets)] * 2

    assert snr_reach(measured, 6.0, 10.0) == pytest.approx(77.6, abs=0.1)
    # Every trace above the limit: no reach; nothing measured: none either.
    assert snr_reach([(offsets, np.full(offsets.size, 20.0))], 6.0, 10.0) is None
    assert snr_reach([], 6.0, 10.0) is None


def test_each_traces_snr_falls_where_its_wave_ends() -> None:
    # The far two thirds of the line carry only noise.
    shot = _shot()
    rng = np.random.default_rng(1)
    xt = shot.xt.copy()
    xt[8:] = (rng.standard_normal(xt[8:].shape) * 0.02).astype(np.float32)

    measured = trace_snrs(replace(shot, xt=xt), VMIN, VMAX, PAD)

    assert measured is not None
    offsets, snrs = measured
    np.testing.assert_allclose(offsets, shot.acquisition.offsets)
    assert snrs[:8].min() > 10 > snrs[8:].max()


def test_a_trace_with_a_dead_band_or_a_gain_of_its_own_stands_out_of_its_neighbours() -> None:
    # White noise on every trace; trace 5 carries 10 dB more, trace 12 nothing between 20 and 40
    # Hz (a notch), both against their neighbours on each side.
    rng = np.random.default_rng(1)
    xt = rng.standard_normal((N_TRACES, 20_000))
    xt[5] *= 10 ** (10 / 20)
    spectrum = np.fft.rfft(xt[12])
    fs = np.fft.rfftfreq(xt.shape[1], d=1 / SAMPLING)
    spectrum[(fs >= 20) & (fs <= 40)] = 0
    xt[12] = np.fft.irfft(spectrum, n=xt.shape[1])

    deviation, dropped = spectral_deviations(xt, SAMPLING, (5.0, 100.0), 2, drop_db=15.0)

    assert deviation[5] == pytest.approx(10.0, abs=1.0)
    assert dropped[12] == pytest.approx(20 / 95, abs=0.03)
    others = np.delete(np.arange(N_TRACES), [5, 12])
    assert np.nanmax(deviation[others]) < 3.0 and np.nanmax(dropped[others]) == 0.0


def test_one_definition_measures_a_shot_and_says_what_each_measure_covers() -> None:
    # PACo's G1 and PAC alike: each measure with its object and what it covers.
    report = measure_signal(_shot(), SignalLimits(), reach_m=12.0)

    measures = {one.name: one for one in report.measures}
    assert list(measures)[:4] == ["dead_traces", "clipped_traces", "nan_traces", "rms_outliers"]
    snr = measures["snr_db"]
    assert snr.of == "signal" and snr.passed and snr.threshold == 6.0
    # Within the reach: the receivers 2 to 12 m from the shot, 11 of the 24; in its band.
    assert snr.over.startswith("11 of 24 traces, within 12 m of the source; in its usable band, ")
    # The record long enough for every trace: its noise after the farthest one's slowest wave,
    # as before the reach (only a record too short for them all takes the reach's).
    assert snr.over.endswith("; noise after the slowest arrival, 0.36 s on")
    assert measures["usable_band_hz"].of == "spectrum"
    assert report.band is not None and report.band[0] < FREQUENCY < report.band[1]


def test_the_snr_and_the_coherence_are_measured_in_the_usable_band() -> None:
    # A 20 Hz wave under a 150 Hz hum, as loud in the noise window as in the wave's: over the
    # whole spectrum its SNR fails; in its band (a filter to it would give the same, and a
    # filter common to the traces changes no image) it passes.
    shot = _shot(noise=0.02)
    hum = 0.5 * np.sin(2 * np.pi * 150.0 * np.asarray(shot.ts, dtype=float))
    humming = replace(shot, xt=(shot.xt + hum[None, :]).astype(np.float32))

    report = measure_signal(humming, SignalLimits())

    measures = {one.name: one for one in report.measures}
    snr = measures["snr_db"]
    assert report.band is not None and report.band[1] < 150.0
    assert np.median(snr_db(humming.xt.astype(float), _windows(humming))) < 6.0
    assert snr.passed and snr.value is not None and snr.value > 6.0
    assert "in its usable band" in snr.over
    assert measures["lateral_coherence"].over.endswith(", in its usable band")


def test_the_snr_and_the_coherence_are_measured_where_the_images_look() -> None:
    # The usable band (0 to 44.5 Hz here) within the band the dispersion images use: its part
    # under 30 Hz; a record whose band misses theirs has nothing they could use.
    report = measure_signal(_shot(), SignalLimits(), image_band=(0.0, 30.0))

    measures = {one.name: one for one in report.measures}
    assert report.band is not None and report.band[1] > 30.0
    assert report.imaged == (report.band[0], 30.0)
    assert f"in its usable band within the images', {report.band[0]:g}-30 Hz" in (
        measures["snr_db"].over
    )
    assert measures["lateral_coherence"].over.endswith("in its usable band within the images'")
    missed = measure_signal(_shot(), SignalLimits(), image_band=(200.0, 300.0))
    band = {one.name: one for one in missed.measures}["usable_band_hz"]
    assert missed.imaged is None and not band.passed
    assert band.over.endswith("; reaching the images' band, 200-300 Hz")


def test_a_band_of_a_frequency_or_two_is_no_band() -> None:
    # Narrower than min_band_hz: no dispersion to see, and no band to measure the SNR in.
    report = measure_signal(_shot(), SignalLimits(min_band_hz=500.0))

    measures = {one.name: one for one in report.measures}
    band = measures["usable_band_hz"]
    assert report.band is None and not band.passed and band.threshold == 500.0
    assert band.value is not None and 0 < band.value < 500.0
    assert "its whole spectrum, no usable band" in measures["snr_db"].over


def test_a_record_is_long_enough_when_it_leaves_its_reach_room_for_noise() -> None:
    # 0.5 s: the slowest wave reaches the farthest trace (25 m) at 0.36 s, too late to leave a
    # noise window after it; within 10 m of the shot, at 0.18 s: room.
    short = _shot(duration_s=0.5)

    assert measure_signal(short, SignalLimits()).windows is None
    assert measure_signal(short, SignalLimits(), reach_m=10.0).windows is not None


def test_a_muted_records_trigger_and_pulse_are_measured_before_its_muting() -> None:
    # A shot 30 ms late in its file, the muting's trigger left at 0: on the muted record its
    # first breaks would be the mute's edge; before its muting, they put the shot 30 ms off.
    late = _shot(t0=0.03)
    muted = Mute(method="mute", vmin=VMIN, vmax=VMAX, width=0.2).transform([late])[0]

    report = measure_signal(muted, SignalLimits(), applied_s=0.0, before_muting=(late, 0.0))

    measures = {one.name: one for one in report.measures}
    trigger = measures["trigger_error_s"]
    assert trigger.value == pytest.approx(0.03, abs=0.005)
    assert not trigger.passed and trigger.threshold == 0.01
    assert trigger.over.endswith(", before the muting")
    assert measures["pulse_s"].value == pytest.approx(0.12, abs=0.02)
    # Without the record before its muting, neither is measured.
    alone = {one.name: one for one in measure_signal(muted, SignalLimits(), applied_s=0.0).measures}
    assert alone["trigger_error_s"].value is None and "pulse_s" not in alone
    # Unmuted, the trigger is reported alone: a common delay changes no image.
    unmuted = {one.name: one for one in measure_signal(late, SignalLimits()).measures}
    error = unmuted["trigger_error_s"]
    assert error.value == pytest.approx(0.03, abs=0.005)
    assert error.passed and error.threshold is None


def test_the_trigger_is_fitted_on_the_direct_wave_nearest_the_shot() -> None:
    # The direct wave at 400 m/s runs through the shot; past 8.2 m a refractor's head wave at
    # 1,500 m/s arrives first, its line crossing the shot's time at 15 ms. Every break fitted
    # reads a trigger 15 ms off; the six nearest, none.
    offsets = np.arange(1.0, 49.0, 1.0)
    breaks = np.minimum(offsets / 400.0, 0.015 + offsets / 1500.0)

    everything = trigger_shift(breaks, offsets)
    nearest = trigger_shift(breaks, offsets, nearest=6)

    assert everything is not None and everything[0] > 0.01
    assert nearest is not None and nearest[0] == pytest.approx(0.0, abs=0.001)
    assert nearest[1] == pytest.approx(400.0, rel=0.01)


def test_a_virtual_shot_is_measured_as_a_shot_against_the_one_limit() -> None:
    # Stacked correlations: their source a receiver, its own trace aside, each lag scaled for
    # the samples it sums; no trigger or pulse to measure; one SNR limit for every signal.
    shot = _shot()
    receivers = shot.acquisition.receivers
    virtual = replace(shot, acquisition=LinearAcquisition(source=receivers[0], receivers=receivers))

    report = measure_signal(virtual, SignalLimits(), source="virtual")

    measures = {one.name: one for one in report.measures}
    assert "trigger_error_s" not in measures and "pulse_s" not in measures
    snr = measures["snr_db"]
    assert snr.threshold == SignalLimits().min_snr_db == 6.0 and snr.passed
    assert snr.over.startswith(
        "23 of 24 traces, the virtual source's own aside, each lag scaled for the samples it sums"
    )
    # Its records muted before correlating, without their correlations before the muting: its
    # noise window zeroed, its SNR and band not measured (hundreds of dB whatever the data).
    zeroed = measure_signal(virtual, SignalLimits(), source="virtual", records_muted=True)
    said = {one.name: one for one in zeroed.measures}
    assert said["snr_db"].value is None and said["snr_db"].passed
    assert said["snr_db"].over.startswith("not measured: its records muted before correlating")
    assert said["usable_band_hz"].value is None and zeroed.band is None


def test_noise_alone_correlated_reads_under_the_limit() -> None:
    # Shots of noise alone, correlated and stacked as a passive-active window's: unscaled, their
    # early lags read 4 to 5 dB over their late ones (a causal correlation sums fewer samples
    # the later its lag); scaled, about 2 dB, under the 6 dB every signal is held to.
    rng = np.random.default_rng(3)
    shots = [
        replace(one, xt=rng.standard_normal(one.xt.shape).astype(np.float32))
        for one in (_shot(seed=k) for k in range(10))
    ]
    chain = Apodize(method="hanning", frac=0.1).transform(shots)
    stacked = Stack(method="linear").transform(
        ActiveShotCorrelation(method="cross").transform(chain)
    )[0]

    report = measure_signal(stacked, SignalLimits(), source="virtual")

    assert report.median_snr is not None and report.median_snr < 4.0
