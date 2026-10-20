from typing import Literal

import numpy as np
from scipy.fft import irfft, rfft
from scipy.fftpack import next_fast_len

from sigpipe.base.stream import Stream


def correlate_cross(
    stream: Stream,
    virtual_source_index: int,
    part: Literal["causal", "acausal", "both"] = "both",
) -> tuple[Stream] | tuple[Stream, Stream]:

    if not (-1 <= virtual_source_index < stream.nx):
        raise ValueError(
            f"virtual_source_index {virtual_source_index} outside [0, {stream.nx - 1}]"
        )

    if part.lower() not in {"causal", "acausal", "both"}:
        raise ValueError(f"part must be 'causal', 'acausal' or 'both, got {part.lower()!r}")

    ts_out = np.arange(stream.nt) / stream.sampling_freq
    # Every trace against the virtual source's at once, through their spectra: scipy's
    # correlate(trace, source, "full") at each lag, the causal lags 0 to nt-1 and the acausal 0
    # to -(nt-1), padded so that no lag wraps onto another.
    nt = stream.nt
    n = next_fast_len(2 * nt - 1)
    spectra = np.asarray(rfft(np.asarray(stream.xt, dtype=np.float64), n=n, axis=1))
    full = np.asarray(irfft(spectra * np.conj(spectra[virtual_source_index])[None, :], n=n, axis=1))
    xt_out_causal = full[:, :nt].astype(np.float32)
    xt_out_acausal = np.concatenate((full[:, :1], full[:, : n - nt : -1]), axis=1).astype(
        np.float32
    )

    acquisition_out = type(stream.acquisition)(
        source=stream.acquisition.receivers[virtual_source_index],
        receivers=stream.acquisition.receivers,
    )

    if part.lower() == "causal":
        return (
            Stream(
                xt=xt_out_causal,
                ts=ts_out,
                sampling_freq=stream.sampling_freq,
                acquisition=acquisition_out,
            ),
        )
    if part.lower() == "acausal":
        return (
            Stream(
                xt=xt_out_acausal,
                ts=ts_out,
                sampling_freq=stream.sampling_freq,
                acquisition=acquisition_out,
            ),
        )
    return Stream(
        xt=xt_out_causal,
        ts=ts_out,
        sampling_freq=stream.sampling_freq,
        acquisition=acquisition_out,
    ), Stream(
        xt=xt_out_acausal,
        ts=ts_out,
        sampling_freq=stream.sampling_freq,
        acquisition=acquisition_out,
    )
