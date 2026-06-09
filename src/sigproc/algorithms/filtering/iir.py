from scipy.signal import butter, sosfiltfilt

from sigproc.base.stream import Stream


def filter_iir(
    stream: Stream,
    *,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> Stream:
    """
    Bandpass filter applied to all traces.
    """
    nyq = 0.5 * stream.sampling_freq
    if not (0 < fmin < fmax < nyq):
        raise ValueError("Require 0 < fmin < fmax < sampling_freq/2")
    sos = butter(order, [fmin / nyq, fmax / nyq], btype="band", output="sos")
    filtered = sosfiltfilt(sos, stream.xt, axis=-1)
    return Stream(
        xt=filtered,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
