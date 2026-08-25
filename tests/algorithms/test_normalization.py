import numpy as np

from sigpipe.algorithms.normalization.onebit import normalize_onebit
from sigpipe.base.stream import Stream


def test_onebit_produces_only_sign_values(stream: Stream) -> None:
    out = normalize_onebit(stream)
    assert set(np.unique(out.xt)).issubset({-1.0, 0.0, 1.0})


def test_onebit_preserves_sign(stream: Stream) -> None:
    out = normalize_onebit(stream)
    np.testing.assert_array_equal(np.sign(stream.xt), out.xt)


def test_onebit_preserves_metadata(stream: Stream) -> None:
    out = normalize_onebit(stream)
    np.testing.assert_array_equal(out.ts, stream.ts)
    assert out.acquisition == stream.acquisition
    assert out.sampling_freq == stream.sampling_freq
