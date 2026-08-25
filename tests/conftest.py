import numpy as np
import pytest

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream


@pytest.fixture
def linear_acquisition() -> LinearAcquisition:
    return LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=tuple(Coordinate(float(i), 0.0, 0.0) for i in range(1, 5)),
    )


@pytest.fixture
def stream(linear_acquisition: LinearAcquisition) -> Stream:
    sampling_freq = 100.0
    nt = 50
    rng = np.random.default_rng(0)
    xt = rng.standard_normal((len(linear_acquisition.receivers), nt))
    ts = np.arange(nt) / sampling_freq
    return Stream(
        xt=xt,
        ts=ts,
        sampling_freq=sampling_freq,
        acquisition=linear_acquisition,
    )
