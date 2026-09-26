from pathlib import Path

import numpy as np
import pytest

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream
from sigpipe.dataio.stream.loading import load_stream
from sigpipe.dataio.stream.saving import save_stream


@pytest.fixture
def saved(tmp_path: Path, stream: Stream) -> list[Path]:
    paths = [tmp_path / "first.hdf5", tmp_path / "second.hdf5"]
    for path in paths:
        save_stream(stream, path)
    return paths


def test_the_same_receivers_of_every_stream(saved: list[Path], stream: Stream) -> None:
    streams = load_stream(saved, receivers_to_load=[1, 3])
    for out in streams:
        np.testing.assert_array_equal(out.xt, stream.xt[[1, 3]])
        assert out.acquisition.receivers == tuple(stream.acquisition.receivers[i] for i in (1, 3))
        assert out.acquisition.source == stream.acquisition.source


def test_each_stream_its_own_receivers(saved: list[Path], stream: Stream) -> None:
    first, second = load_stream(saved, receivers_to_load=[[0, 1, 2], [1, 2]])
    np.testing.assert_array_equal(first.xt, stream.xt[[0, 1, 2]])
    np.testing.assert_array_equal(second.xt, stream.xt[[1, 2]])


def test_given_acquisitions_replace_the_saved_ones(saved: list[Path], stream: Stream) -> None:
    receivers = stream.acquisition.receivers[:2]
    acquisition = LinearAcquisition(source=Coordinate(-5.0, 0.0, 0.0), receivers=receivers)
    streams = load_stream(saved, acquisitions=[acquisition] * 2, receivers_to_load=[0, 1])
    assert all(out.acquisition == acquisition for out in streams)
    with pytest.raises(ValueError, match="number of receivers"):
        load_stream(saved, acquisitions=[acquisition] * 2)


def test_receivers_for_each_file_must_match_the_files(saved: list[Path]) -> None:
    with pytest.raises(ValueError, match="one receivers_to_load per file"):
        load_stream(saved, receivers_to_load=[[0, 1]])
    with pytest.raises(TypeError, match="receivers_to_load"):
        load_stream(saved, receivers_to_load=["a"])  # type: ignore[list-item]
