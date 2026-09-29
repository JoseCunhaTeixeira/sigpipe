"""Measures of a record's signal (PACo's signal QC, G1, judges them): dead, clipped and NaN
traces; amplitudes off the decay with offset; the SNR and the usable band, from a surface-wave
window against a noise window; how far from the shot the traces carry the wave; lateral
coherence; the first breaks, the trigger they point to and the shot's pulse after them; each
trace's spectrum against its neighbours' (a noise record's too)."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, hilbert, welch
from scipy.stats import theilslopes

from sigpipe.base.stream import Stream


@dataclass(frozen=True)
class Windows:
    """The samples of each trace in the surface-wave window and in the noise window."""

    signal: np.ndarray  # bool, (n_traces, n_samples)
    noise: np.ndarray  # bool, (n_traces, n_samples)
    where: str  # which noise window was used


def signal_windows(
    offsets: np.ndarray,
    ts: np.ndarray,
    vmin: float,
    vmax: float,
    pad: float,
    within: np.ndarray | None = None,
) -> Windows | None:
    """The surface-wave window of each trace, between the arrivals at `vmax` and `vmin` (m/s)
    widened by `pad` seconds on both sides, and a noise window: before the trigger when the
    record has samples there, else after the slowest arrival of every trace when the record is
    long enough, else after the slowest arrival of the traces `within` the reach (beyond it no
    trace carries the wave, and none is measured: a long line's record is long enough for its
    reach); None when it is not."""
    starts = offsets / vmax - pad
    stops = offsets / vmin + pad
    signal = (ts[None, :] >= starts[:, None]) & (ts[None, :] <= stops[:, None])
    if ts[0] < -pad:
        noise = np.broadcast_to(ts < 0.0, signal.shape)
        return Windows(signal, np.array(noise), "before the trigger")
    every = np.ones(offsets.size, dtype=bool)
    reached = every
    for traces in (every, within):
        if traces is None or not traces.any():
            continue
        reached = traces
        if ts[-1] - float(stops[traces].max()) >= (stops[traces] - starts[traces]).mean():
            break
    else:
        return None
    after = float(stops[reached].max())
    noise = np.broadcast_to(ts > after, signal.shape)
    return Windows(signal, np.array(noise), f"after the slowest arrival, {after:.2f} s on")


def dead_clipped_nan(
    xt: np.ndarray, dead_ratio: float, clip_share: float
) -> tuple[np.ndarray, ...]:
    """Masks of the dead traces (RMS below `dead_ratio` of the median), the clipped ones (more
    than `clip_share` of their samples at their extreme) and those holding NaN."""
    nan = np.asarray(np.isnan(xt).any(axis=1))
    finite = np.nan_to_num(xt)
    rms = np.sqrt(np.mean(finite**2, axis=1))
    dead = rms < dead_ratio * np.median(rms[~nan]) if (~nan).any() else rms == 0
    peak = np.abs(finite).max(axis=1, keepdims=True)
    at_peak = np.abs(finite) >= (1 - 1e-3) * np.where(peak > 0, peak, 1.0)
    clipped = (at_peak.mean(axis=1) > clip_share) & ~dead
    return dead, clipped, nan


def rms_decay_outliers(
    rms: np.ndarray, offsets: np.ndarray, usable: np.ndarray, mads: float, factor: float = 2.0
) -> np.ndarray:
    """Traces whose RMS is far from the fit of log RMS against log offset (the decay with
    distance): by more than `mads` median absolute deviations, and by at least `factor`; among
    the usable traces."""
    outliers = np.zeros(rms.size, dtype=bool)
    fit_on = usable & (rms > 0) & (offsets > 0)
    if fit_on.sum() < 4:
        return outliers
    x, y = np.log(offsets[fit_on]), np.log(rms[fit_on])
    fit = np.asarray(theilslopes(y, x), dtype=float)  # robust: one bad trace does not tilt it
    slope, intercept = fit[0], fit[1]
    residuals = y - (slope * x + intercept)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    limit = max(mads * 1.4826 * mad, math.log(factor))
    outliers[fit_on] = np.abs(residuals - np.median(residuals)) > limit
    return outliers


def trace_snrs(
    stream: Stream, vmin: float, vmax: float, pad: float, shot_s: float = 0.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Each trace's distance from the shot and its SNR in dB (surface-wave window against noise
    window, see signal_windows), its times from the shot at `shot_s` (a record whose time
    origin was left where its file puts the trigger); None when the record leaves no room for a
    noise window."""
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    ts = np.asarray(stream.ts, dtype=float) - shot_s
    windows = signal_windows(offsets, ts, vmin, vmax, pad)
    if windows is None:
        return None
    return offsets, snr_db(np.nan_to_num(stream.xt), windows)


def snr_reach(
    measured: Sequence[tuple[np.ndarray, np.ndarray]], min_db: float, bin_m: float
) -> float | None:
    """How far from the shot the traces still carry the wave: over every record's traces
    `measured` (offsets, SNR), binned every `bin_m` m from the shot, the distance where the
    bins' median SNR first falls below `min_db`, between the centres of the last bin above and
    the first below; None when no bin falls below."""
    if not measured or bin_m <= 0:
        return None
    offsets = np.concatenate([one[0] for one in measured])
    snrs = np.concatenate([one[1] for one in measured])
    bins = np.floor(offsets / bin_m).astype(int)
    previous: tuple[float, float] | None = None
    for index in range(int(bins.max()) + 1):
        values = snrs[bins == index]
        if values.size == 0:
            continue
        centre, median = (index + 0.5) * bin_m, float(np.median(values))
        if median < min_db:
            if previous is None:
                return round(centre, 2)
            before, above = previous
            return round(before + (above - min_db) / (above - median) * (centre - before), 2)
        previous = (centre, median)
    return None


def snr_db(xt: np.ndarray, windows: Windows) -> np.ndarray:
    """Each trace's energy per sample in the signal window over the noise window, in dB."""
    signal = energy_per_sample(xt, windows.signal)
    noise = energy_per_sample(xt, windows.noise)
    return 10 * np.log10((signal + 1e-30) / (noise + 1e-30))


def usable_band(
    xt: np.ndarray,
    sampling_freq: float,
    windows: Windows,
    band_db: float,
    peak_db: float = 20.0,
) -> tuple[float, float] | None:
    """The band of frequencies, around the signal spectrum's peak, where the signal windows'
    spectrum exceeds the noise windows' by `band_db` and stays within `peak_db` of its peak
    among them (a hum's line, as loud in the noise, sets no peak); None when no frequency
    does."""
    spectra = window_spectra(xt, windows, sampling_freq)
    if spectra is None:
        return None
    freqs, signal_power, noise_power = spectra
    above = 10 * np.log10((signal_power + 1e-30) / (noise_power + 1e-30)) > band_db
    if not above.any():
        return None
    peak_power = float(signal_power[above].max())
    above &= 10 * np.log10((signal_power + 1e-30) / (peak_power + 1e-30)) > -peak_db
    peak = int(np.argmax(np.where(above, signal_power, -np.inf)))
    low = peak
    while low > 0 and above[low - 1]:
        low -= 1
    high = peak
    while high < above.size - 1 and above[high + 1]:
        high += 1
    return float(freqs[low]), float(freqs[high])


def lateral_coherence(
    xt: np.ndarray, windows: Windows, sampling_freq: float, max_lag_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """For each pair of neighbouring traces, the peak of their normalized cross-correlation in
    the signal window within `max_lag_s`, and its sign: the coherence, and the polarity."""
    max_lag = max(1, round(max_lag_s * sampling_freq))
    peaks = np.zeros(xt.shape[0] - 1)
    for i in range(xt.shape[0] - 1):
        mask = windows.signal[i] | windows.signal[i + 1]
        # In double precision: squared sums of large samples (a correlation's) overflow single.
        a, b = xt[i][mask].astype(float), xt[i + 1][mask].astype(float)
        norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if norm == 0:
            continue
        full = correlate(a, b, mode="full") / norm
        centre = a.size - 1
        window = full[max(0, centre - max_lag) : centre + max_lag + 1]
        peaks[i] = window[np.argmax(np.abs(window))]
    return np.abs(peaks), np.sign(peaks)


def _envelope(xt: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Each trace's envelope, smoothed over 5 ms."""
    span = max(3, round(0.005 / max(float(ts[1] - ts[0]), 1e-9)))
    return uniform_filter1d(np.abs(np.asarray(hilbert(xt, axis=1))), span, axis=1)


def first_breaks(xt: np.ndarray, ts: np.ndarray, windows: Windows, ratio: float) -> np.ndarray:
    """Each trace's first break: where its envelope, smoothed over 5 ms, first rises above
    `ratio` times its noise RMS; NaN when it never does."""
    noise_rms = np.sqrt(energy_per_sample(xt, windows.noise))
    envelope = _envelope(xt, ts)
    breaks = np.full(xt.shape[0], np.nan)
    for i, trace in enumerate(envelope):
        above = np.flatnonzero(trace > ratio * noise_rms[i])
        if above.size:
            breaks[i] = ts[above[0]]
    return breaks


def pulse_durations(
    xt: np.ndarray, ts: np.ndarray, breaks: np.ndarray, ratio: float, longest_s: float
) -> np.ndarray:
    """Each trace's pulse, how long its energy lasts from its first break: to where its envelope,
    smoothed over 5 ms, falls back below `ratio` times the peak it reaches within `longest_s` of
    the break. NaN without a break, or when it has not fallen back by then."""
    envelope = _envelope(xt, ts)
    pulses = np.full(xt.shape[0], np.nan)
    for i, (trace, start) in enumerate(zip(envelope, breaks, strict=True)):
        if np.isnan(start):
            continue
        after = np.flatnonzero((ts >= start) & (ts <= start + longest_s))
        if after.size < 2:
            continue
        peak = after[np.argmax(trace[after])]
        fallen = after[(after > peak) & (trace[after] < ratio * trace[peak])]
        if fallen.size:
            pulses[i] = float(ts[fallen[0]] - start)
    return pulses


def trigger_shift(
    breaks: np.ndarray, offsets: np.ndarray, nearest: int | None = None
) -> tuple[float, float, float] | None:
    """The first breaks fitted as t = t0 + offset / v, robustly (Theil-Sen): (t0, v, the
    breaks' scatter about the line, as 1.4826 x their median absolute deviation); None with
    fewer than 4 breaks. With `nearest`, that many breaks nearest the shot alone: the direct
    wave's, whose line runs through the shot (farther, a refractor's head wave arrives first,
    its line crossing the shot's time at its intercept time, no trigger's). Without, every
    break, spanning half the offsets at least (a noisy far half fakes a shift)."""
    valid = ~np.isnan(breaks)
    if nearest is not None:
        chosen = [i for i in np.argsort(offsets) if valid[i]][:nearest]
        valid = np.zeros(breaks.size, dtype=bool)
        valid[chosen] = True
    if valid.sum() < 4:
        return None
    span = offsets[valid].max() - offsets[valid].min()
    if nearest is None and span < 0.5 * (offsets.max() - offsets.min()):
        return None
    fit = np.asarray(theilslopes(breaks[valid], offsets[valid]), dtype=float)
    slowness, t0 = float(fit[0]), float(fit[1])
    if slowness <= 0:
        return None
    residuals = breaks[valid] - (t0 + slowness * offsets[valid])
    scatter = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    return float(t0), float(1 / slowness), scatter


def energy_per_sample(xt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = mask.sum(axis=1)
    energy = np.sum(np.where(mask, xt, 0.0) ** 2, axis=1)
    return np.where(counts > 0, energy / np.maximum(counts, 1), 0.0)


def window_spectra(
    xt: np.ndarray, windows: Windows, sampling_freq: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The traces' mean power spectra in their surface-wave window and in their noise window,
    per sample, at one resolution: each trace's noise window cut in pieces as long as its
    surface-wave window (half overlapping), each piece and the window tapered (Hann) and
    zero-padded to the record's length, the pieces averaged. A short window cannot resolve a
    line (a hum's): its lobe spreads over 2 / its length on each side; compared with the long
    noise window's sharp line, its sides would seem signal. None without a trace holding both."""
    n = xt.shape[1]
    signal_power = np.zeros(n // 2 + 1)
    noise_power = np.zeros(n // 2 + 1)
    counted = 0
    for trace, in_signal, in_noise in zip(xt, windows.signal, windows.noise, strict=True):
        signal_at = np.flatnonzero(in_signal)
        noise_at = np.flatnonzero(in_noise)
        length = min(signal_at.size, noise_at.size)
        if signal_at.size < 2 or length < 2:
            continue
        taper = np.hanning(signal_at.size)
        signal_power += _tapered_power(trace[signal_at], taper, n)
        # A noise window shorter than the surface-wave window (before an early trigger): one
        # piece of it, as long as it is.
        piece = np.hanning(length)
        starts = range(0, noise_at.size - length + 1, max(1, length // 2))
        noise_power += np.mean(
            [_tapered_power(trace[noise_at[k : k + length]], piece, n) for k in starts], axis=0
        )
        counted += 1
    if counted == 0:
        return None
    freqs = np.fft.rfftfreq(n, d=1 / sampling_freq)
    return freqs, signal_power / counted, noise_power / counted


def _tapered_power(samples: np.ndarray, taper: np.ndarray, n: int) -> np.ndarray:
    """`samples` tapered and zero-padded to `n`: their power spectrum per sample."""
    return np.abs(np.fft.rfft(samples * taper, n=n)) ** 2 / float(np.sum(taper**2))


def spectral_deviations(
    xt: np.ndarray,
    sampling_freq: float,
    band: tuple[float, float],
    neighbours: int,
    drop_db: float,
    usable: np.ndarray | None = None,
    shape: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Each trace's power spectrum (Welch's, 1 s pieces) against the median of its `neighbours`
    on each side that are `usable`, over `band` (Hz): how far it sits from theirs, the median of
    |10 log10(its / theirs)| (dB: a gain or a response of its own), and the share of the band
    where it falls more than `drop_db` below theirs (a dead band). With `shape`, each spectrum's
    own level taken out first: a shot's traces, louder the nearer, compared by their shapes.
    NaN for a trace not `usable`, with no usable neighbour, or no frequency in `band`."""
    n_traces = xt.shape[0]
    usable = np.ones(n_traces, dtype=bool) if usable is None else usable
    nperseg = min(xt.shape[1], max(8, round(sampling_freq)))
    fs, power = welch(xt, fs=sampling_freq, nperseg=nperseg, axis=1)
    inside = (fs >= band[0]) & (fs <= band[1])
    deviation = np.full(n_traces, np.nan)
    dropped = np.full(n_traces, np.nan)
    if not inside.any():
        return deviation, dropped
    level = 10 * np.log10(np.maximum(power[:, inside], 1e-30))
    if shape:
        level -= np.median(level, axis=1, keepdims=True)
    for i in range(n_traces):
        if not usable[i]:
            continue
        around = [
            j
            for j in range(max(0, i - neighbours), min(n_traces, i + neighbours + 1))
            if j != i and usable[j]
        ]
        if not around:
            continue
        difference = level[i] - np.median(level[around], axis=0)
        deviation[i] = float(np.median(np.abs(difference)))
        dropped[i] = float(np.mean(difference < -drop_db))
    return deviation, dropped
