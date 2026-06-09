from scipy.signal import detrend

from sigproc.base.stream import Stream


def detrend_constant(stream: Stream) -> Stream:
    xt_detrended = detrend(stream.xt, axis=1, type="constant")
    return Stream(
        xt=xt_detrended,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=stream.acquisition,
    )
