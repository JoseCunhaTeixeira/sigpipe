"""The surface waves' cone on shot gathers, and the energy arriving faster than it.

A shot's surface waves carry most of its energy: each trace's envelope peaks on them. The
peaks' apparent velocities (offset over peak time, from the shot) spread over the velocities
the waves cross the line at; their interquartile range is the cone a mute keeps, widened by the
caller. Energy arriving before the cone's fast edge is body waves or refractions, which a mute
around the cone takes out of the dispersion image. Traces nearer the shot than a few receiver
spacings are left out: their peaks sit in the near field, at the source's own delay."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from sigpipe.base.stream import Stream

# The fewest traces a cone is drawn from: fewer, their quartiles say little.
MIN_TRACES = 8


@dataclass(frozen=True)
class Cone:
    """The surface waves' velocities, m/s: the slow and fast edges of the traces' envelope peaks
    (their lower and upper quartiles), their median (which a minority of fast arrivals, such as
    strong refractions, leaves where the surface waves are), and how many traces gave them."""

    vmin: float
    vmax: float
    median: float
    n_traces: int


def peak_velocities(stream: Stream, shot_s: float, min_offset_m: float) -> np.ndarray:
    """Each trace's apparent velocity, m/s: its offset over the time of its envelope's peak after
    the shot (`shot_s`, on the record's time axis). Traces nearer the shot than `min_offset_m`,
    and those whose peak is not after the shot, are left out."""
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    times = np.asarray(stream.ts, dtype=float) - shot_s
    after = times > 0
    if not after.any():
        return np.empty(0)
    envelope = np.abs(hilbert(np.asarray(stream.xt, dtype=float), axis=-1))
    peaks = times[np.argmax(np.where(after, envelope, -np.inf), axis=-1)]
    kept = (offsets >= min_offset_m) & (peaks > 0)
    return offsets[kept] / peaks[kept]


def surface_wave_cone(velocities: Iterable[np.ndarray]) -> Cone | None:
    """The cone of the gathers whose traces' apparent velocities are `velocities` (one array a
    gather): the interquartile range over every trace. None from fewer than MIN_TRACES."""
    pooled = np.concatenate([np.asarray(one, dtype=float) for one in velocities] or [np.empty(0)])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < MIN_TRACES:
        return None
    low, middle, high = np.percentile(pooled, (25, 50, 75))
    return Cone(
        vmin=round(float(low), 1),
        vmax=round(float(high), 1),
        median=round(float(middle), 1),
        n_traces=int(pooled.size),
    )


def fast_share(
    stream: Stream,
    shot_s: float,
    vmax: float,
    slowest: float,
    pad_s: float,
    min_offset_m: float,
) -> float | None:
    """The share of the gather's signal energy that arrives faster than `vmax`: from the shot to
    offset / `vmax`, over the signal's, from the shot to offset / `slowest` + `pad_s`; traces
    nearer the shot than `min_offset_m` left out. None when no trace has signal."""
    offsets = np.asarray(stream.acquisition.offsets, dtype=float)
    times = np.asarray(stream.ts, dtype=float) - shot_s
    energy = np.asarray(stream.xt, dtype=float) ** 2
    kept = offsets >= min_offset_m
    if not kept.any():
        return None
    after = times[None, :] > 0
    signal = after & (times[None, :] <= offsets[kept, None] / slowest + pad_s)
    early = after & (times[None, :] < offsets[kept, None] / vmax)
    total = float((energy[kept] * signal).sum())
    if total <= 0:
        return None
    return round(float((energy[kept] * early).sum()) / total, 4)
