import numpy as np
import pytest

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.arrivals import TraceArrivals
from sigpipe.base.stream import Stream


def test_nt_and_nx(stream: Stream) -> None:
    assert stream.nx == 4
    assert stream.nt == 50


def test_arrays_are_read_only(stream: Stream) -> None:
    with pytest.raises(ValueError, match="read-only"):
        stream.xt[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        stream.ts[0] = 1.0


def test_xt_and_ts_cast_to_float32(stream: Stream) -> None:
    assert stream.xt.dtype == np.float32
    assert stream.ts.dtype == np.float32


def test_sampling_freq_cast_to_float(linear_acquisition: LinearAcquisition) -> None:
    s = Stream(
        xt=np.zeros((4, 10)),
        ts=np.arange(10),
        sampling_freq=100,  # type: ignore[arg-type]
        acquisition=linear_acquisition,
    )
    assert s.sampling_freq == 100.0
    assert isinstance(s.sampling_freq, float)


def test_xt_must_be_2d(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(ValueError, match="xt must be 2D"):
        Stream(
            xt=np.zeros(10),
            ts=np.arange(10),
            sampling_freq=100.0,
            acquisition=linear_acquisition,
        )


def test_ts_must_be_1d(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(ValueError, match="ts must be 1D"):
        Stream(
            xt=np.zeros((4, 10)),
            ts=np.zeros((10, 1)),
            sampling_freq=100.0,
            acquisition=linear_acquisition,
        )


def test_nt_ts_mismatch_raises(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(ValueError, match="nt and ts mismatch"):
        Stream(
            xt=np.zeros((4, 10)),
            ts=np.arange(5),
            sampling_freq=100.0,
            acquisition=linear_acquisition,
        )


def test_nx_receivers_mismatch_raises(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(ValueError, match="xt and receivers mismatch"):
        Stream(
            xt=np.zeros((3, 10)),
            ts=np.arange(10),
            sampling_freq=100.0,
            acquisition=linear_acquisition,
        )


def test_arrivals_must_be_tuple(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(TypeError, match="TraceArrivals"):
        Stream(
            xt=np.zeros((4, 10)),
            ts=np.arange(10),
            sampling_freq=100.0,
            acquisition=linear_acquisition,
            arrivals=[TraceArrivals()] * 4,  # type: ignore[arg-type]
        )


def test_arrivals_count_must_match_nx(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(ValueError, match="Expected 4 arrivals"):
        Stream(
            xt=np.zeros((4, 10)),
            ts=np.arange(10),
            sampling_freq=100.0,
            acquisition=linear_acquisition,
            arrivals=(TraceArrivals(),) * 3,
        )


def test_arrivals_elements_must_be_trace_arrivals(linear_acquisition: LinearAcquisition) -> None:
    with pytest.raises(TypeError, match="All arrivals must be TraceArrivals"):
        Stream(
            xt=np.zeros((4, 10)),
            ts=np.arange(10),
            sampling_freq=100.0,
            acquisition=linear_acquisition,
            arrivals=(TraceArrivals(), TraceArrivals(), TraceArrivals(), "not-arrivals"),  # type: ignore[arg-type]
        )


def test_xt_analytic_envelope_and_phase_shapes(stream: Stream) -> None:
    assert stream.xt_analytic.shape == stream.xt.shape
    assert stream.xt_envelope.shape == stream.xt.shape
    assert stream.xt_phase.shape == stream.xt.shape
    assert stream.xt_envelope.dtype == np.float32
    assert stream.xt_phase.dtype == np.float32


def test_xt_envelope_is_non_negative(stream: Stream) -> None:
    assert np.all(stream.xt_envelope >= 0)


def test_hilbert_envelope_of_pure_cosine_is_constant(
    linear_acquisition: LinearAcquisition,
) -> None:
    sampling_freq = 100.0
    nt = 256
    ts = np.arange(nt) / sampling_freq
    freq = 5.0
    amplitude = 3.0
    trace = amplitude * np.cos(2 * np.pi * freq * ts)
    xt = np.tile(trace, (len(linear_acquisition.receivers), 1))

    s = Stream(xt=xt, ts=ts, sampling_freq=sampling_freq, acquisition=linear_acquisition)

    edge = 64
    envelope = s.xt_envelope[:, edge:-edge]
    np.testing.assert_allclose(envelope, amplitude, atol=0.1)
