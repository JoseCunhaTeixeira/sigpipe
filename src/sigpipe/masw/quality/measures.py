"""The measures of a signal, one definition for the assistant's checks (PACo's G1 on a record, G2
on a window's stacked correlations) and PAC's views alike: each measure with the object it
describes (`of`: the signal, or its spectrum) and what it covers (`over`: which traces, which
window), judged against its limit. A record's with a shot, and a virtual shot's (the stacked
correlations, their source a receiver), alike; a noise record's, its traces only.

What a dispersion image is made of decides what is measured (2026-09-29): the phase shift
divides each trace's spectrum by its own amplitude, so a filter common to the traces cannot
change the image, nor a common delay (the trigger) when nothing is muted. The SNR and the
coherence are measured in the signal's usable band (the frequencies where its surface waves
stand over its noise), the noise and the arrivals on the record before its muting."""

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import butter, sosfiltfilt

from sigpipe.base.stream import Stream
from sigpipe.dataio.selection_plotting import SelectionScores
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

# What a signal is measured from: a shot at a known time (a record), a virtual shot (stacked
# correlations: their source a receiver, its own trace aside), or none (a noise record).
Source = Literal["shot", "virtual", "none"]


class Measure(BaseModel):
    """One measurement against its limit: the value must stay above it (min) or below it (max);
    without one, reported only. `of`: the object it describes (signal, spectrum, ...); `over`:
    what it covers ("52 of 96 traces, within 63.35 m of the shot")."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float | None  # None: could not be measured
    threshold: float | None = None
    bound: Literal["min", "max"] | None = None
    passed: bool
    unit: str = ""
    of: str = ""
    over: str = ""


class SignalLimits(BaseModel):
    """How a signal is measured, and its limits: provisional, measured on the demo profiles."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    vg_min: float = Field(default=80.0, gt=0, description="m/s, the slowest surface wave kept")
    vg_max: float = Field(default=1500.0, gt=0, description="m/s, the fastest arrival kept")
    pad_s: float = Field(default=0.05, ge=0, description="s, added on both sides of the window")
    pulse_ratio: float = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="The shot's pulse ends where its envelope falls back below this share of its "
        "peak",
    )
    pulse_traces: int = Field(
        default=3, ge=1, description="Traces nearest the shot whose pulses give the record's"
    )
    max_pulse_s: float = Field(
        default=0.5, gt=0, description="s, the longest pulse searched: longer, not measured"
    )
    # A noise record's traces, and a shot's, against their neighbours' spectra (the user,
    # 2026-09-28): flagged, never left out. On the demo, a passive trace sits 1 dB from its
    # neighbours (3.4 at the 99th percentile); a shot's, its own level taken out and the five
    # nearest the shot aside, 1.7 (5.6).
    spectra_fmin_hz: float = Field(
        default=2.0, ge=0, description="Hz, where the traces' spectra are compared from"
    )
    spectra_fmax_share: float = Field(
        default=0.8, gt=0, le=1, description="Share of Nyquist the spectra are compared up to"
    )
    spectra_neighbours: int = Field(
        default=2, ge=1, description="Traces on each side a trace's spectrum is compared with"
    )
    max_spectral_deviation_db: float = Field(
        default=6.0,
        gt=0,
        description="dB, a trace's spectrum's median distance from its neighbours', at most: "
        "beyond, a gain or a response of its own",
    )
    dead_band_db: float = Field(
        default=15.0, gt=0, description="dB under the neighbours' that makes a frequency dead"
    )
    max_dead_band_share: float = Field(
        default=0.1, gt=0, le=1, description="Share of the band a trace may have dead, at most"
    )
    spectra_near_traces: int = Field(
        default=5,
        ge=0,
        description="Traces nearest a shot left out of the comparison: louder and brighter by "
        "their distance alone",
    )
    dead_ratio: float = Field(
        default=0.01,
        gt=0,
        description="A trace whose RMS is below this share of the median is dead.",
    )
    clip_share: float = Field(
        default=0.005,
        gt=0,
        description="Share of samples at a trace's extreme that means clipping.",
    )
    rms_outlier_mads: float = Field(
        default=3.0, gt=0, description="RMS off the decay with offset by this many MADs, in log."
    )
    rms_outlier_factor: float = Field(
        default=2.0,
        gt=1,
        description="...and by at least this factor: a smooth decay has tiny MADs.",
    )
    min_snr_db: float = Field(
        default=6.0,
        description="dB, the traces' median SNR in their usable band: a record's, and a window's "
        "stacked correlations' measured as a shot's from their virtual source (one limit for "
        "every signal, the user, 2026-09-29).",
    )
    reach_snr_db: float = Field(
        default=2.0,
        description="dB: the traces' median SNR, by distance from the shot, under which they carry "
        "no wave: the line's reach (line_reach), the traces a record's measures cover.",
    )
    min_fk_kept_share: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="Share of a passive window's segments its fk selection keeps, at least.",
    )
    band_db: float = Field(
        default=6.0, gt=0, description="Signal above noise, for the usable band."
    )
    min_band_hz: float = Field(
        default=2.0,
        gt=0,
        description="Hz, the usable band's width at least: narrower, a frequency or two, no "
        "dispersion to see and no band to measure the SNR in.",
    )
    peak_db: float = Field(
        default=20.0,
        gt=0,
        description="...and within this much of the signal spectrum's peak: after the wave, the\n"
        "noise window is quieter than the signal window at every frequency.",
    )
    min_coherence: float = Field(
        default=0.5, ge=0, le=1, description="Median correlation of neighbouring traces."
    )
    max_trigger_error_s: float = Field(
        default=0.01,
        ge=0,
        description="s, the first breaks' shot time off where the muting's trigger put it: "
        "judged on a muted record alone (unmuted, a common delay changes no image).",
    )
    trigger_traces: int = Field(
        default=6,
        ge=4,
        description="The first breaks nearest the shot the trigger is fitted on: the direct "
        "wave's (farther, a refractor's head wave arrives first, its intercept time no "
        "trigger's).",
    )
    max_trigger_scatter_s: float = Field(
        default=0.05,
        ge=0,
        description="s, the first breaks' scatter about their line: beyond it they are no line, "
        "and no correction fixes the record.",
    )
    first_break_ratio: float = Field(
        default=5.0, gt=0, description="A first break is the first sample above this x noise RMS."
    )


@dataclass(frozen=True)
class SignalReport:
    """A signal's measures, and what its checks act on: its bad traces, its windows, the
    traces measured, its usable band, its first breaks' fit."""

    measures: tuple[Measure, ...]
    dead: np.ndarray  # bool, by trace: dead, clipped, NaN, left out (excluded already)
    clipped: np.ndarray
    nan: np.ndarray
    left_out: np.ndarray
    windows: Windows | None = None  # the surface waves' and the noise's; None: no room for noise
    noise_where: str | None = None  # where the noise was measured, "…, before the muting" too
    usable: np.ndarray | None = None  # the traces its SNR, band and coherence may use
    median_snr: float | None = None
    band: tuple[float, float] | None = None  # its usable band, the whole of it
    imaged: tuple[float, float] | None = None  # the part of it the images use, its SNR's
    median_coherence: float | None = None
    # The direct wave's line through the first breaks nearest the shot: t0 (the trigger's
    # error), v, their scatter; and the scatter of every break about a line through them all.
    fit: tuple[float, float, float] | None = None
    scatter: float | None = None
    pulse: float | None = None


def measure_signal(
    stream: Stream,
    limits: SignalLimits,
    *,
    source: Source = "shot",
    shot_s: float = 0.0,
    applied_s: float | None = None,
    excluded: Collection[int] = (),
    reach_m: float | None = None,
    before_muting: tuple[Stream, float] | None = None,
    spectra: bool = False,
    records_muted: bool = False,
    image_band: tuple[float, float] | None = None,
) -> SignalReport:
    """`stream`'s measures against `limits`. Its traces `excluded` already (receiver indices)
    are left out of every measure; the decay with offset, the SNR, the usable band and the
    lateral coherence are measured on the traces within `reach_m` of the source (the line's
    reach, beyond which the traces carry no wave; a virtual source's own trace aside), and the
    record is long enough when it leaves them room for a noise window. Times from the source at
    `shot_s`; `applied_s`, the shift the muting applied (None: off). The SNR and the coherence
    are measured in the usable band (as a filter to it would, which changes no image), within
    the band the dispersion images use (`image_band`: the dispersion's fmin and fmax); the
    noise, the usable band, the first breaks and the shot's pulse on the record before its
    muting (`before_muting`: it, and where its shot is on it), when muted. With `spectra`, the
    traces off their neighbours' spectra, reported. A virtual shot (stacked correlations) is
    measured as a shot, each lag scaled for the samples it sums; `records_muted`: its records
    muted before they were correlated, its SNR and band not measured without the correlations
    of the records before their muting (`before_muting`)."""
    xt = stream.xt
    n_traces = xt.shape[0]
    left_out = _left_out(n_traces, excluded)
    dead, clipped, nan = (
        mask & ~left_out for mask in dead_clipped_nan(xt, limits.dead_ratio, limits.clip_share)
    )
    bad = dead | clipped | nan | left_out
    considered = f"{_traces(int((~left_out).sum()), n_traces)}" + (
        f" ({int(left_out.sum())} left out)" if left_out.any() else ""
    )
    measures = [
        Measure(
            name=name,
            value=int(mask.sum()),
            threshold=0,
            bound="max",
            passed=not mask.any(),
            of="signal",
            over=considered,
        )
        for name, mask in (("dead_traces", dead), ("clipped_traces", clipped), ("nan_traces", nan))
    ]
    if spectra:
        off, judged = spectral_outliers(
            stream, limits, excluded, shots=source != "none", near=_near_traces(source, limits)
        )
        band_hz = (limits.spectra_fmin_hz, limits.spectra_fmax_share * stream.sampling_freq / 2)
        measures.append(
            Measure(
                name="spectral_outliers",
                value=int(off.sum()),
                passed=True,
                of="spectrum",
                over=f"{_traces(int(judged.sum()), n_traces)}, {band_hz[0]:g}-{band_hz[1]:g} Hz"
                + (", their shapes" if source != "none" else ""),
            )
        )

    if source == "none":
        return SignalReport(tuple(measures), dead, clipped, nan, left_out)
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    within = _within(offsets, reach_m)
    if source == "virtual":
        within &= offsets > 0  # the virtual source's own trace: its autocorrelation
    # Times from the shot where it should be: its windows on its arrivals, muted or not.
    ts = np.asarray(stream.ts, dtype=float) - shot_s
    windows = signal_windows(offsets, ts, limits.vg_min, limits.vg_max, limits.pad_s, within)
    if windows is None:
        return SignalReport(tuple(measures), dead, clipped, nan, left_out)

    # In double precision: a correlation's samples are products of traces, and their squares'
    # sums overflow a record's single precision.
    scale = _lags_scaled if source == "virtual" else _as_is
    finite = scale(np.nan_to_num(np.asarray(xt, dtype=float)))
    outliers = np.zeros(n_traces, dtype=bool)
    if source == "shot":
        # Reported, not left out: a trace off the decay in this record alone (the one nearest
        # the shot, where the fitted decay overshoots; a burst of noise) is no bad geophone.
        # The line judges each receiver over every record (judge_receivers). The record's own
        # measures leave them out.
        rms = np.sqrt(np.mean(finite**2, axis=1))
        outliers, fitted = _off_decay(
            rms, offsets, ~(dead | clipped | nan), within, left_out, limits
        )
        measures.append(
            Measure(
                name="rms_outliers",
                value=int(outliers.sum()),
                passed=True,
                of="signal",
                over=f"{_traces(int(fitted.sum()), n_traces)} alive" + _reach(reach_m, within),
            )
        )
    usable = ~(bad | outliers)
    measured = usable & within if (usable & within).any() else usable
    unmuted = _unmuted(before_muting, finite, ts, windows, offsets, limits, within, scale)
    noise = unmuted.windows
    # A virtual shot of muted records without their correlations before the muting: its noise
    # window zeroed, its SNR hundreds of dB whatever the data (the user, 2026-09-29).
    zeroed = source == "virtual" and records_muted and not unmuted.muted
    over = f"{_traces(int(measured.sum()), n_traces)}" + _reach(reach_m, within)
    if source == "virtual":
        over += ", the virtual source's own aside, each lag scaled for the samples it sums"
    noise_where = noise.where + (", before the muting" if unmuted.muted else "")
    found = (
        usable_band(
            unmuted.xt[measured],
            stream.sampling_freq,
            Windows(noise.signal[measured], noise.noise[measured], noise.where),
            limits.band_db,
            limits.peak_db,
        )
        if measured.any() and not zeroed
        else None
    )
    band = found if found is not None and found[1] - found[0] >= limits.min_band_hz else None
    # The part of it the dispersion images use (the user, 2026-09-29): above it, a record's
    # near-source energy no image uses, where neighbouring traces agree less.
    imaged = _imaged(band, image_band, limits.min_band_hz)
    # The SNR and the coherence in that band, as a filter to it would give them: a filter common
    # to the traces changes no dispersion image, so neither may it change what G1 decides.
    snr = snr_db(_band_passed(unmuted.xt, stream.sampling_freq, imaged), noise)
    median_snr = float(np.median(snr[measured])) if measured.any() else float("nan")
    snr_ok = zeroed or median_snr >= limits.min_snr_db
    within = " within the images'" if image_band is not None else ""
    in_band = (
        f"in its usable band{within}, {imaged[0]:g}-{imaged[1]:g} Hz"
        if imaged is not None
        else f"its whole spectrum, no usable band{within}"
    )
    not_measured = "not measured: its records muted before correlating, their noise zeroed"
    measures.append(
        Measure(
            name="snr_db",
            value=None if zeroed else _finite(median_snr),
            threshold=limits.min_snr_db,
            bound="min",
            passed=snr_ok,
            unit="dB",
            of="signal",
            over=not_measured if zeroed else f"{over}; {in_band}; noise {noise_where}",
        )
    )
    measures.append(
        Measure(
            name="usable_band_hz",
            value=None if found is None else _finite(found[1] - found[0]),
            threshold=limits.min_band_hz,
            bound="min",
            passed=zeroed or imaged is not None,
            unit="Hz",
            of="spectrum",
            over=not_measured
            if zeroed
            else f"{over}; {limits.band_db:g} dB over the noise {noise_where}"
            + (
                f"; reaching the images' band, {image_band[0]:g}-{image_band[1]:g} Hz"
                if image_band is not None
                else ""
            ),
        )
    )
    max_lag_s = float(np.diff(np.sort(offsets)).min()) / limits.vg_min if offsets.size > 1 else 0.0
    coherence, _ = lateral_coherence(
        _band_passed(finite, stream.sampling_freq, imaged), windows, stream.sampling_freq, max_lag_s
    )
    # Within the reach too: pairs of noise traces beyond it are not the record's fault.
    pair_ok = measured[:-1] & measured[1:]
    median_coherence = float(np.median(coherence[pair_ok])) if pair_ok.any() else float("nan")
    measures.append(
        Measure(
            name="lateral_coherence",
            value=_finite(median_coherence),
            threshold=limits.min_coherence,
            bound="min",
            passed=median_coherence >= limits.min_coherence,
            of="signal",
            over=f"{int(pair_ok.sum())} pairs of neighbouring traces, in the surface-wave window"
            + (f", in its usable band{within}" if imaged is not None else ""),
        )
    )
    if source == "virtual":
        return SignalReport(
            tuple(measures),
            dead,
            clipped,
            nan,
            left_out,
            windows=windows,
            noise_where=noise_where,
            usable=usable,
            median_snr=None if zeroed else _finite(median_snr),
            band=band,
            imaged=imaged,
            median_coherence=_finite(median_coherence),
        )

    # The arrivals on the record before its muting: a mute zeroes what comes before the fastest
    # arrival, and a first break on it is the mute's edge.
    arrivals = unmuted.muted or applied_s is None
    breaks = (
        first_breaks(unmuted.xt, unmuted.ts, noise, limits.first_break_ratio)
        if arrivals
        else np.full(n_traces, np.nan)
    )
    # Only the traces whose own SNR passes: a noisy trace's envelope crosses the threshold on
    # noise, early or late.
    breaks[~usable | (snr < limits.min_snr_db)] = np.nan
    # The shot's pulse (the user, 2026-09-28): how long its energy lasts after the first break,
    # the median over the traces nearest the shot that show one; the width a mute keeps after
    # the slowest arrival.
    pulse: float | None = None
    if arrivals:
        pulses = pulse_durations(
            unmuted.xt, unmuted.ts, breaks, limits.pulse_ratio, limits.max_pulse_s
        )
        nearest = [i for i in np.argsort(offsets) if np.isfinite(pulses[i])]
        found_pulse = float(np.median(pulses[nearest[: limits.pulse_traces]])) if nearest else None
        pulse = _finite(found_pulse)
        measures.append(
            Measure(
                name="pulse_s",
                value=pulse,
                passed=True,
                unit="s",
                of="signal",
                over=f"the {min(len(nearest), limits.pulse_traces)} traces nearest the shot "
                "showing one" + (", before the muting" if unmuted.muted else ""),
            )
        )
    # The trigger from the breaks nearest the shot: the direct wave's line runs through the
    # shot (farther, a refractor's head wave arrives first, its intercept no trigger's). Their
    # scatter about a line through every break: per-trace delays, which no correction fixes.
    # Without four breaks, or on a noisy record whose breaks come late, neither is measured,
    # nor wrong.
    near = trigger_shift(breaks, offsets, nearest=limits.trigger_traces) if snr_ok else None
    every = trigger_shift(breaks, offsets) if snr_ok else None
    shift = None if near is None else near[0]
    scatter = None if every is None else every[2]
    found = int(np.isfinite(breaks).sum())
    before = ", before the muting" if unmuted.muted else ""
    error_over = (
        f"the first breaks of the {min(found, limits.trigger_traces)} traces nearest the shot "
        f"whose SNR passes{before}"
    )
    # The trigger is the muting's: unmuted, a common delay changes no image, reported.
    muted = applied_s is not None
    measures += [
        Measure(
            name="trigger_error_s",
            value=_finite(shift),
            threshold=limits.max_trigger_error_s if muted else None,
            bound="max" if muted else None,
            passed=not muted or shift is None or abs(shift) <= limits.max_trigger_error_s,
            unit="s",
            of="signal",
            over=error_over,
        ),
        Measure(
            name="trigger_scatter_s",
            value=_finite(scatter),
            threshold=limits.max_trigger_scatter_s,
            bound="max",
            passed=scatter is None or scatter <= limits.max_trigger_scatter_s,
            unit="s",
            of="signal",
            over=f"the first breaks of the {found} traces whose SNR passes{before}",
        ),
    ]
    return SignalReport(
        tuple(measures),
        dead,
        clipped,
        nan,
        left_out,
        windows=windows,
        noise_where=noise_where,
        usable=usable,
        median_snr=_finite(median_snr),
        band=band,
        imaged=imaged,
        median_coherence=_finite(median_coherence),
        fit=near,
        scatter=scatter,
        pulse=pulse,
    )


def line_reach(
    records: Iterable[tuple[Stream, float]], limits: SignalLimits, span_m: float
) -> float | None:
    """How far from the shots the line's traces still carry the wave: each trace's SNR over
    every record (`records`: each before its muting, as the SNR is measured, and where its shot
    is on it), binned every fifteenth of the line (`span_m`), where their median falls under
    `reach_snr_db` (snr_reach); None when it never does. PACo's G1 and PAC measure the records
    within it alike."""
    measured = [
        found
        for stream, shot_s in records
        if (found := trace_snrs(stream, limits.vg_min, limits.vg_max, limits.pad_s, shot_s))
        is not None
    ]
    return snr_reach(measured, limits.reach_snr_db, span_m / 15) if span_m > 0 else None


def selection_measures(selection: SelectionScores, limits: SignalLimits) -> tuple[Measure, ...]:
    """A passive window's fk segment selection (as its job saved it), measured: the segments it
    judged, the share it kept (at least min_fk_kept_share), those it flipped."""
    over = f"the window's {selection.segments} segments, |f-k ratio| > {selection.threshold:g}"
    return (
        Measure(
            name="fk_segments", value=selection.segments, passed=True, of="selection", over=over
        ),
        Measure(
            name="fk_kept",
            value=round(selection.kept_share, 4),
            threshold=limits.min_fk_kept_share,
            bound="min",
            passed=selection.kept_share >= limits.min_fk_kept_share,
            of="selection",
            over=over,
        ),
        Measure(
            name="fk_flipped",
            value=selection.flipped_count,
            passed=True,
            of="selection",
            over=over,
        ),
    )


def decay_outliers(
    stream: Stream,
    limits: SignalLimits,
    excluded: Collection[int] = (),
    reach_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The record's traces off the decay with offset (receiver indices as a mask), and those
    judged for it: alive, within `reach_m` of the shot, not `excluded`."""
    xt = stream.xt
    left_out = _left_out(xt.shape[0], excluded)
    dead, clipped, nan = dead_clipped_nan(xt, limits.dead_ratio, limits.clip_share)
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    rms = np.sqrt(np.mean(np.nan_to_num(xt) ** 2, axis=1))
    return _off_decay(
        rms, offsets, ~(dead | clipped | nan), _within(offsets, reach_m), left_out, limits
    )


def spectral_outliers(
    stream: Stream,
    limits: SignalLimits,
    excluded: Collection[int] = (),
    shots: bool = False,
    near: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The traces off their neighbours' spectra (a gain or a response of their own, a dead
    band), and those judged: a noise record's by level and shape, a shot's by shape alone with
    the `near` traces nearest its shot aside (its limits' by default); the traces `excluded`,
    dead, clipped or NaN neither judged nor compared with."""
    xt = stream.xt
    dead, clipped, nan = dead_clipped_nan(xt, limits.dead_ratio, limits.clip_share)
    usable = ~(dead | clipped | nan | _left_out(xt.shape[0], excluded))
    if shots:
        offsets = np.asarray(stream.acquisition.offsets, dtype=float)
        aside = limits.spectra_near_traces if near is None else near
        usable[np.argsort(offsets)[:aside]] = False
    band = (limits.spectra_fmin_hz, limits.spectra_fmax_share * stream.sampling_freq / 2)
    deviation, dropped = spectral_deviations(
        np.nan_to_num(np.asarray(xt, dtype=float)),
        stream.sampling_freq,
        band,
        limits.spectra_neighbours,
        limits.dead_band_db,
        usable,
        shape=shots,
    )
    judged = np.isfinite(deviation)
    off = judged & (
        (deviation > limits.max_spectral_deviation_db) | (dropped > limits.max_dead_band_share)
    )
    return off, judged


def _near_traces(source: Source, limits: SignalLimits) -> int:
    """The traces nearest a source its spectra comparison leaves aside: a shot's, louder and
    brighter by their distance alone; a virtual source's own, its autocorrelation."""
    return 1 if source == "virtual" else limits.spectra_near_traces


@dataclass(frozen=True)
class _Unmuted:
    """The traces a signal's noise, usable band and arrivals are measured on: its own, or, when
    muted, the record's before its muting (its times from its shot)."""

    xt: np.ndarray
    ts: np.ndarray
    windows: Windows
    muted: bool  # the record before its muting's


def _unmuted(
    before_muting: tuple[Stream, float] | None,
    finite: np.ndarray,
    ts: np.ndarray,
    windows: Windows,
    offsets: np.ndarray,
    limits: SignalLimits,
    within: np.ndarray,
    scale: Callable[[np.ndarray], np.ndarray],
) -> _Unmuted:
    """The record before its muting (`before_muting`: it, and where its shot is on it), scaled
    as the signal is; the signal itself (`finite`, `ts`, `windows`) when not muted, or when the
    record before its muting leaves no room for a noise window."""
    if before_muting is None:
        return _Unmuted(finite, ts, windows, muted=False)
    stream, shot_s = before_muting
    its_ts = np.asarray(stream.ts, dtype=float) - shot_s
    found = signal_windows(offsets, its_ts, limits.vg_min, limits.vg_max, limits.pad_s, within)
    if found is None or stream.xt.shape != finite.shape:
        return _Unmuted(finite, ts, windows, muted=False)
    return _Unmuted(
        scale(np.nan_to_num(np.asarray(stream.xt, dtype=float))), its_ts, found, muted=True
    )


def _imaged(
    band: tuple[float, float] | None, image_band: tuple[float, float] | None, min_width: float
) -> tuple[float, float] | None:
    """The part of usable `band` within `image_band` (the band the dispersion images use), when
    at least `min_width` wide; `band` itself without an image band."""
    if band is None or image_band is None:
        return band
    low, high = max(band[0], image_band[0]), min(band[1], image_band[1])
    return (low, high) if high - low >= min_width else None


def _as_is(xt: np.ndarray) -> np.ndarray:
    return xt


def _lags_scaled(xt: np.ndarray) -> np.ndarray:
    """Stacked causal correlations, each lag scaled by the square root of the share of samples
    it sums (n - k of the n correlated, at lag k): noise alone then as strong at every lag, as
    on a record, where unscaled it reads 4 to 5 dB louder early than late (the user,
    2026-09-29)."""
    n = xt.shape[1]
    return xt / np.sqrt((n - np.arange(n)) / n)[None, :]


def _band_passed(
    xt: np.ndarray, sampling_freq: float, band: tuple[float, float] | None
) -> np.ndarray:
    """`xt` filtered to `band` as the preprocessing's IIR filter does (a 4th-order Butterworth,
    run both ways), its top kept under Nyquist as a preset's (0.95 of it); left whole without a
    band, or with a band over the whole spectrum."""
    if band is None:
        return xt
    nyquist = sampling_freq / 2
    low, high = band[0], min(band[1], 0.95 * nyquist)
    top = high >= 0.95 * nyquist
    if (low <= 0 and top) or low >= high:
        return xt
    if low <= 0:
        sos = butter(4, high / nyquist, btype="low", output="sos")
    elif top:
        sos = butter(4, low / nyquist, btype="high", output="sos")
    else:
        sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    return np.asarray(sosfiltfilt(sos, xt, axis=-1))


def _off_decay(
    rms: np.ndarray,
    offsets: np.ndarray,
    alive: np.ndarray,
    within: np.ndarray,
    left_out: np.ndarray,
    limits: SignalLimits,
) -> tuple[np.ndarray, np.ndarray]:
    """The traces off the decay with offset, and those judged. The decay is fitted on every
    trace `alive` within the reach, those already left out among them: excluding a trace must
    not move the fit, or each round excludes more. Within the reach only: over the whole line
    the noise floor flattens the fit, and the traces nearest the shots, the strongest, read too
    loud."""
    fit_on = alive & within
    outliers = (
        rms_decay_outliers(rms, offsets, fit_on, limits.rms_outlier_mads, limits.rms_outlier_factor)
        & ~left_out
        & within
    )
    return outliers, fit_on & ~left_out


def _within(offsets: np.ndarray, reach_m: float | None) -> np.ndarray:
    """The traces within the line's reach: beyond it they carry no wave, and no window stacks
    them."""
    if reach_m is not None and (offsets <= reach_m).any():
        return offsets <= reach_m
    return np.ones(offsets.size, dtype=bool)


def _left_out(n_traces: int, excluded: Collection[int]) -> np.ndarray:
    left_out = np.zeros(n_traces, dtype=bool)
    left_out[[index for index in excluded if 0 <= index < n_traces]] = True
    return left_out


def _traces(count: int, total: int) -> str:
    return f"{count} of {total} traces" if count != total else f"all {total} traces"


def _reach(reach_m: float | None, within: np.ndarray) -> str:
    """The reach, when it leaves some traces out."""
    return (
        f", within {reach_m:g} m of the source" if reach_m is not None and not within.all() else ""
    )


def _finite(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 4)
