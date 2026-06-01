import numpy as np
from scipy.signal import correlate


def correlate_cross(
    xt: np.ndarray,
    ts: np.ndarray,
    sampling_freq: float,
    *,
    virtual_source_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if xt.ndim != 2:
        raise ValueError("xt must have shape [nx, nt]")

    nx, nt = xt.shape

    if len(ts) != nt:
        raise ValueError(f"ts length ({len(ts)}) must match nt ({nt})")

    if sampling_freq <= 0:
        raise ValueError("sampling_freq must be > 0")

    if not (0 <= virtual_source_index < xt.shape[0]):
        raise ValueError(
            f"virtual_source_index={virtual_source_index} "
            f"outside [0, {xt.shape[0] - 1}]"
        )

    ts_out = np.arange(nt) / sampling_freq
    xt_out_causal = np.zeros((nx, len(ts_out)), dtype=np.float32)
    xt_out_acausal = np.zeros((nx, len(ts_out)), dtype=np.float32)

    trace_source = xt[virtual_source_index, :]
    for i_receiver, trace_receiver in enumerate(xt):
        correl = correlate(trace_source, trace_receiver, mode="full")
        mid = correl.size // 2
        causal = correl[mid:]
        acausal = correl[: mid + 1][::-1]
        if len(causal) != len(ts_out) or len(acausal) != len(ts_out):
            raise ValueError(
                f"causal and acaucal lengths must be the same as ts_out, got {len(causal)} and {len(acausal)}"
            )
        xt_out_causal[i_receiver, :] = causal
        xt_out_acausal[i_receiver, :] = acausal

    return (
        ts_out.astype(np.float32),
        xt_out_causal.astype(np.float32),
        xt_out_acausal.astype(np.float32),
    )
