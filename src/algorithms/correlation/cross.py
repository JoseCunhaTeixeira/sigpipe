import numpy as np
from scipy.signal import correlate

from src.base.acquisition import Acquisition
from src.base.stream import Stream


def correlate_cross(
    stream: Stream,
    virtual_source_index: int,
) -> tuple[Stream, Stream]:

    if not (0 <= virtual_source_index < stream.nx):
        raise ValueError(
            f"virtual_source_index {virtual_source_index} outside [0, {stream.nx - 1}]"
        )

    ts_out = np.arange(stream.nt) / stream.sampling_freq
    xt_out_causal = np.zeros((stream.nx, len(ts_out)), dtype=np.float32)
    xt_out_acausal = np.zeros((stream.nx, len(ts_out)), dtype=np.float32)

    trace_source = stream.xt[virtual_source_index, :]
    for i_receiver, trace_receiver in enumerate(stream.xt):
        correl = correlate(trace_source, trace_receiver, mode="full")
        mid = correl.size // 2
        causal = correl[mid:]
        acausal = correl[: mid + 1][::-1]
        if len(causal) != len(ts_out) or len(acausal) != len(ts_out):
            raise ValueError(
                f"causal and acaucal lengths must be the same as ts_out, got {len(causal)}, {len(acausal)} and {len(ts_out)}"
            )
        xt_out_causal[i_receiver, :] = causal
        xt_out_acausal[i_receiver, :] = acausal

    acquisition_out = Acquisition(
        source=stream.acquisition.receivers[virtual_source_index],
        receivers=stream.acquisition.receivers,
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
