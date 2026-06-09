import numpy as np

from sigproc.base.stream import Stream


def _cosine_taper(n: int) -> np.ndarray:
    """Raised cosine taper from 0 to 1."""
    if n <= 0:
        return np.empty(0)

    x = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(x))


def _apply_lower_mute(trace: np.ndarray, idx: int, taper: int) -> None:
    """
    Mute everything before idx.

    Result:
        0 0 0 | taper | 1 1 1
    """
    n = trace.size

    idx = np.clip(idx, 0, n)

    if taper <= 0:
        trace[:idx] = 0.0
        return

    start = max(0, idx - taper)

    trace[:start] = 0.0

    if start < idx:
        w = _cosine_taper(idx - start)
        trace[start:idx] *= w


def _apply_upper_mute(trace: np.ndarray, idx: int, taper: int) -> None:
    """
    Mute everything after idx.

    Result:
        1 1 1 | taper | 0 0 0
    """
    n = trace.size

    idx = np.clip(idx, 0, n)

    if taper <= 0:
        trace[idx:] = 0.0
        return

    stop = min(n, idx + taper)

    if idx < stop:
        w = _cosine_taper(stop - idx)[::-1]
        trace[idx:stop] *= w

    trace[stop:] = 0.0


def mute(
    stream: Stream,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    taper: int = 0,
) -> Stream:
    """
    Squared-window mute with optional cosine taper.
    """
    if tmin is None and tmax is None and vmin is None and vmax is None:
        raise ValueError("At least one of tmin, tmax, vmin and vmax must be provided")

    if tmin is not None and tmax is not None and tmin >= tmax:
        raise ValueError(f"Expected tmin < tmax, got tmin={tmin} and tmax={tmax}")

    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError(f"Expected vmin < vmax, got vmin={vmin} and vmax={vmax}")

    xt = stream.xt.copy()
    ts = stream.ts

    if tmin is not None:
        idx = int(np.searchsorted(ts, tmin))

        for trace in xt:
            _apply_lower_mute(trace, idx, taper)

    if tmax is not None:
        idx = int(np.searchsorted(ts, tmax))

        for trace in xt:
            _apply_upper_mute(trace, idx, taper)

    if vmin is not None:
        tlims = stream.acquisition.offsets / vmin

        for i_trace, tlim in enumerate(tlims):
            idx = np.searchsorted(ts, tlim)
            _apply_upper_mute(xt[i_trace], idx, taper)

    if vmax is not None:
        tlims = stream.acquisition.offsets / vmax

        for i_trace, tlim in enumerate(tlims):
            idx = np.searchsorted(ts, tlim)
            _apply_lower_mute(xt[i_trace], idx, taper)

    return Stream(
        xt=xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
