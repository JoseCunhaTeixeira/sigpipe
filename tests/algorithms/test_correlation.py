"""The cross-correlation: every trace against the virtual source's, as scipy's correlate gives
them, at the causal and the acausal lags."""

import numpy as np
import pytest
from scipy.signal import correlate

from sigpipe.algorithms.correlation.cross import correlate_cross
from sigpipe.base.stream import Stream


@pytest.mark.parametrize("virtual_source_index", [0, 2, -1])
def test_each_trace_against_the_source_as_scipy_correlates_them(
    stream: Stream, virtual_source_index: int
) -> None:
    made = correlate_cross(stream, virtual_source_index, part="both")
    assert len(made) == 2
    causal, acausal = made[0], made[-1]

    source = stream.xt[virtual_source_index]
    for i, trace in enumerate(stream.xt):
        full = correlate(trace.astype(np.float64), source.astype(np.float64), mode="full")
        mid = full.size // 2
        np.testing.assert_allclose(causal.xt[i], full[mid:], rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(acausal.xt[i], full[: mid + 1][::-1], rtol=1e-5, atol=1e-4)
    assert causal.xt.shape == acausal.xt.shape == stream.xt.shape
    np.testing.assert_allclose(causal.ts, np.arange(stream.nt) / stream.sampling_freq)
