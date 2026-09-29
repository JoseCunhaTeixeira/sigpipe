import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure

from sigpipe.base.stream import Stream
from sigpipe.dataio.plot_config import CM, DISP_DPI, DOUBLE_COLUMN_CM

# The samples a trace keeps, as PAC's gather view does: what a figure shows, far faster.
GATHER_SAMPLES = 1200
# PAC's gather colours: the traces' ink, the shot's star.
_INK = "#1a1a1a"
_SOURCE = "#2a78d6"


def plot_stream(stream: Stream, normalize: bool = True) -> Figure:
    """`stream`'s traces as PAC's gather view draws them: wiggles at their receivers' positions
    along the line (their offsets when the line runs along y), positive lobes filled, each
    trace scaled to its own largest value (`normalize`; else the gather's), the whole of it, at
    most GATHER_SAMPLES samples a trace; time down. A shot off the receivers is a star above its
    position (a virtual source or a passive record's stand-in, on a receiver, none)."""
    full = np.asarray(stream.xt, dtype=np.float32)
    peaks = np.max(np.abs(full), axis=1, keepdims=True) if normalize else np.max(np.abs(full))
    full = full / (peaks + 1e-12)
    stride = max(1, -(-stream.nt // GATHER_SAMPLES))
    xt = full[:, ::stride]
    ts = np.asarray(stream.ts[::stride], dtype=np.float32)
    receivers = stream.acquisition.receivers
    xs = np.array([receiver.x for receiver in receivers], dtype=float)
    if np.ptp(xs) == 0 and xs.size > 1:
        xs = np.asarray(stream.acquisition.offsets, dtype=float)
    gaps = np.diff(np.sort(xs))
    gap = float(np.min(gaps[gaps > 0])) if (gaps > 0).any() else 1.0
    half = 0.9 * gap

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_CM * CM, 11.0 * CM), dpi=DISP_DPI, layout="constrained"
    )
    wiggles = [np.column_stack((x + half * trace, ts)) for x, trace in zip(xs, xt, strict=True)]
    lobes = [
        np.column_stack(
            (
                np.concatenate(([x], x + half * np.maximum(trace, 0.0), [x])),
                np.concatenate(([ts[0]], ts, [ts[-1]])),
            )
        )
        for x, trace in zip(xs, xt, strict=True)
    ]
    ax.add_collection(PolyCollection(lobes, facecolors=_INK, edgecolors="none", linewidths=0))
    ax.add_collection(LineCollection(wiggles, colors=_INK, linewidths=0.35))
    source = stream.acquisition.source.x
    shot = bool(np.isfinite(source)) and not bool(np.any(np.isclose(xs, source)))
    if shot:
        ax.plot([source], [ts[0]], marker="*", color=_SOURCE, markersize=9, clip_on=False)
    if stream.arrivals is not None:
        for arrivals, x in zip(stream.arrivals, xs, strict=False):
            for arrival in arrivals:
                ax.plot([x], [arrival.time], marker="+", color="red", markersize=5)
    low = min(float(xs.min()), source) if shot else float(xs.min())
    high = max(float(xs.max()), source) if shot else float(xs.max())
    ax.set_xlim(low - gap, high + gap)
    ax.set_ylim(float(ts[-1]), float(ts[0]))
    ax.set_xlabel("Position [m]", fontsize=7)
    ax.set_ylabel("Time [s]", fontsize=7)
    ax.tick_params(labelsize=6)
    return fig
