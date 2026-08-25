import numpy as np
import pytest

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.stream import Stream
from sigpipe.transformers import Apodize, Detrend, Flip, Normalize, Pad


def test_detrend_none_is_passthrough(stream: Stream) -> None:
    out = Detrend(method="none").transform([stream])
    assert out == [stream]


def test_detrend_applies_named_algorithm(stream: Stream) -> None:
    out = Detrend(method="constant").transform([stream])
    assert len(out) == 1
    np.testing.assert_allclose(out[0].xt.mean(axis=1), 0.0, atol=1e-4)


def test_detrend_unknown_method_raises(stream: Stream) -> None:
    with pytest.raises(ValueError, match="Unknown detrending method"):
        Detrend(method="bogus").transform([stream])


def test_detrend_rejects_non_stream_elements() -> None:
    with pytest.raises(TypeError):
        Detrend(method="constant").transform([1, 2, 3])  # type: ignore[list-item]


def test_detrend_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="Empty input sequence"):
        Detrend(method="constant").transform([])


def test_pad_extends_every_stream_in_sequence(stream: Stream) -> None:
    out = Pad(n=5).transform([stream, stream])
    assert len(out) == 2
    assert all(s.nt == stream.nt + 5 for s in out)


def test_flip_applies_axis_to_every_stream(stream: Stream) -> None:
    out = Flip(axis="space").transform([stream])
    np.testing.assert_array_equal(out[0].xt, stream.xt[::-1, :])


def test_normalize_none_is_passthrough(stream: Stream) -> None:
    out = Normalize(method="none").transform([stream])
    assert out == [stream]


def test_normalize_onebit_applies_sign(stream: Stream) -> None:
    out = Normalize(method="onebit").transform([stream])
    np.testing.assert_array_equal(out[0].xt, np.sign(stream.xt))


def test_apodize_hanning_tapers_edges(linear_acquisition: LinearAcquisition) -> None:
    nt = 40
    xt = np.ones((len(linear_acquisition.receivers), nt))
    ts = np.arange(nt, dtype=np.float32)
    s = Stream(xt=xt, ts=ts, sampling_freq=1.0, acquisition=linear_acquisition)

    out = Apodize(method="hanning", frac=0.1).transform([s])
    assert out[0].xt[0, 0] == 0.0


def test_pipeline_chains_stream_transformers(stream: Stream) -> None:
    pipeline = Detrend(method="constant") >> Normalize(method="onebit") >> Pad(n=10)
    result = pipeline.run([stream], show_log=False)
    assert len(result) == 1
    assert result[0].nt == stream.nt + 10
    assert set(np.unique(result[0].xt[:, : stream.nt])).issubset({-1.0, 0.0, 1.0})
