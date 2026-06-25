from scipy.signal import butter, sosfiltfilt

from sigpipe.base.stream import Stream


def filter_iir(
    stream: Stream,
    *,
    fmin: float,
    fmax: float,
    order: int = 4,
) -> Stream:

    nyq = 0.5 * stream.sampling_freq

    if not (0 <= fmin < fmax < nyq):
        raise ValueError("Require 0 <= fmin < fmax < sampling_freq/2")

    if fmin == 0:
        sos = butter(
            order,
            fmax / nyq,
            btype="low",
            output="sos",
        )
    else:
        sos = butter(
            order,
            [fmin / nyq, fmax / nyq],
            btype="band",
            output="sos",
        )

    filtered = sosfiltfilt(sos, stream.xt, axis=-1)

    return Stream(
        xt=filtered,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
