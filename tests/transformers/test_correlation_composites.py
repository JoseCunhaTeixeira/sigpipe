import numpy as np
import pytest

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream
from sigpipe.transformers import ActiveShotCorrelation, BidirectionalCorrelate


def _shot(stream: Stream, source_x: float) -> Stream:
    acquisition = LinearAcquisition(
        source=Coordinate(source_x, 0.0, 0.0),
        receivers=stream.acquisition.receivers,
    )
    return Stream(
        xt=stream.xt,
        ts=stream.ts,
        sampling_freq=stream.sampling_freq,
        acquisition=acquisition,
    )


@pytest.mark.parametrize(("source_x", "zero_lag_trace"), [(0.0, 0), (5.0, -1)])
def test_active_shot_gather_is_seen_from_its_first_receiver(
    stream: Stream, source_x: float, zero_lag_trace: int
) -> None:
    (out,) = ActiveShotCorrelation(method="cross").transform([_shot(stream, source_x)])
    assert out.acquisition.source == stream.acquisition.receivers[0]
    assert out.acquisition.receivers == stream.acquisition.receivers
    np.testing.assert_allclose(out.acquisition.offsets, [0.0, 1.0, 2.0, 3.0])
    # The virtual source's own trace, first: its autocorrelation at zero lag.
    energy = float(np.sum(stream.xt[zero_lag_trace].astype(np.float64) ** 2))
    np.testing.assert_allclose(out.xt[0, 0], energy, rtol=1e-4)


def test_active_shot_inside_the_line_is_skipped(stream: Stream) -> None:
    assert ActiveShotCorrelation(method="cross").transform([_shot(stream, 2.5)]) == []


def test_bidirectional_gathers_are_all_seen_from_the_first_receiver(stream: Stream) -> None:
    # Causal and acausal parts, from the first receiver then from the last (flipped).
    outputs = BidirectionalCorrelate(method="cross").transform([stream])
    assert len(outputs) == 4
    for out in outputs:
        assert out.acquisition.source == stream.acquisition.receivers[0]
        np.testing.assert_allclose(out.acquisition.offsets, [0.0, 1.0, 2.0, 3.0])
    for out, virtual_source in zip(outputs, (0, 0, -1, -1), strict=True):
        energy = float(np.sum(stream.xt[virtual_source].astype(np.float64) ** 2))
        np.testing.assert_allclose(out.xt[0, 0], energy, rtol=1e-4)
