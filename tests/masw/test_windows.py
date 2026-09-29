from pathlib import Path

import pytest
from pydantic import ValidationError

from sigpipe.base import Coordinate, LinearAcquisition
from sigpipe.masw.profiles import Profile, ProfileKind, Record
from sigpipe.masw.windows import (
    Exclusions,
    MASWParameters,
    MASWWindow,
    apply_exclusions,
    build_windows,
    nearest_offset,
)

RECEIVERS = tuple(Coordinate(float(x), 0.0, 0.0) for x in range(10))


def _profile(sources: list[float | None]) -> Profile:
    """A line of 10 receivers 1 m apart, a record per source (None: passive)."""
    records = tuple(
        Record(
            path=Path(f"{i}.dat"),
            n_traces=len(RECEIVERS),
            sampling_rate_hz=1000.0,
            duration_s=1.0,
            source=None if x is None else Coordinate(x, 0.0, 0.0),
        )
        for i, x in enumerate(sources)
    )
    kind = ProfileKind.PASSIVE if sources == [None] * len(sources) else ProfileKind.ACTIVE
    return Profile(
        name="line", kind=kind, folder=Path("line"), records=records, receivers=RECEIVERS
    )


def test_windows_slide_along_the_line_with_the_shots_outside_them() -> None:
    # Shots before the line, past it, and one inside it.
    profile = _profile([-1.0, 11.0, 4.5])

    windows = build_windows(profile, MASWParameters(length=5, step=1))

    assert [window.xmid for window in windows] == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    for window in windows:
        receivers = [RECEIVERS[i] for i in window.receiver_indices]
        low, high = receivers[0].x, receivers[-1].x
        inside = [path.name for path in window.selected_files if path.name == "2.dat"]
        assert inside == ([] if low < 4.5 < high else ["2.dat"])
        assert all(one.receivers == tuple(receivers) for one in window.acquisitions)


def test_the_distance_from_the_shot_bounds_the_windows_it_serves() -> None:
    profile = _profile([-1.0])

    windows = build_windows(
        profile, MASWParameters(length=3, step=1, distance_min=2.0, distance_max=5.0)
    )

    # The window middle 2 to 5 m from the shot, both bounds excluded: xmids 2 and 3 m.
    assert [window.xmid for window in windows] == [2.0, 3.0]
    # A bound left out is none: the nearest from 0, the farthest at any distance.
    nearest = build_windows(profile, MASWParameters(length=3, step=1, distance_max=5.0))
    farthest = build_windows(profile, MASWParameters(length=3, step=1, distance_min=2.0))
    every = build_windows(profile, MASWParameters(length=3, step=1))
    assert [window.xmid for window in nearest] == [1.0, 2.0, 3.0]
    assert [window.xmid for window in farthest][:2] == [2.0, 3.0]
    assert every[0].xmid == 1.0 and len(every) == len(farthest) + 1


def test_passive_records_serve_every_window_from_its_first_receiver() -> None:
    windows = build_windows(_profile([None, None]), MASWParameters(length=4, step=3))

    assert [window.xmid for window in windows] == [1.5, 4.5, 7.5]
    for window in windows:
        assert len(window.selected_files) == 2
        first = RECEIVERS[window.receiver_indices[0]]
        assert all(one.source == first for one in window.acquisitions)


def test_a_window_longer_than_the_line_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds the 10 receivers"):
        build_windows(_profile([-1.0]), MASWParameters(length=11))


@pytest.mark.parametrize(
    "overrides",
    [
        {"length": 2},
        {"step": 0},
        {"distance_min": -1.0},
        {"distance_max": 0.0},
        {"distance_min": 5.0, "distance_max": 5.0},
    ],
)
def test_invalid_masw_parameters(overrides: dict[str, float]) -> None:
    valid = {"length": 24, "step": 12, "distance_min": 1.0, "distance_max": 10.0}

    with pytest.raises(ValidationError):
        MASWParameters.model_validate(valid | overrides)


def test_each_record_leaves_out_its_own_traces_on_active_windows() -> None:
    receivers = tuple(Coordinate(float(x), 0.0, 0.0) for x in range(6))

    def acquisition(source: float) -> LinearAcquisition:
        return LinearAcquisition(source=Coordinate(source, 0.0, 0.0), receivers=receivers)

    window = MASWWindow(
        xmid=2.5,
        selected_files=[Path("a.dat"), Path("b.dat"), Path("c.dat")],
        receiver_indices=list(range(6)),
        acquisitions=[acquisition(-1.0), acquisition(-2.0), acquisition(7.0)],
    )
    # Trace 1 is a.dat's own; trace 4 two of the three records excluded: a receiver's defect.
    exclusions = Exclusions(traces={"a.dat": (1, 4), "b.dat": (4,)})

    # One geometry (passive): a trace any record excluded leaves every record.
    union = apply_exclusions(window, exclusions)
    assert union is not None and union.receiver_indices == [0, 2, 3, 5]
    assert union.record_receivers is None
    # Each its own (active): the window keeps its receivers, each record gives its own.
    own = apply_exclusions(window, exclusions, "per_record")
    assert own is not None and own.receiver_indices == list(range(6))
    assert own.record_receivers == [[0, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 5]]
    assert [len(one.receivers) for one in own.acquisitions] == [4, 5, 5]
    # One set of receivers (passive-active, correlation gathers stacked): trace 4 leaves every
    # record, and a.dat, which excluded trace 1 too, leaves the window.
    shared = apply_exclusions(window, exclusions, "shared")
    assert shared is not None and shared.receiver_indices == [0, 1, 2, 3, 5]
    assert [path.name for path in shared.selected_files] == ["b.dat", "c.dat"]
    assert shared.record_receivers is None
    assert [len(one.receivers) for one in shared.acquisitions] == [5, 5]
    # Two records: every exclusion is half of them, so it is the union again.
    two = window.model_copy(
        update={
            "selected_files": window.selected_files[:2],
            "acquisitions": window.acquisitions[:2],
        }
    )
    both = apply_exclusions(two, Exclusions(traces={"a.dat": (1,)}), "per_record")
    assert both is not None and both.record_receivers == [[0, 2, 3, 4, 5]] * 2


def test_the_nearest_offset_is_from_the_nearest_shot_to_the_nearest_receiver() -> None:
    receivers = tuple(Coordinate(float(x), 0.0, 0.0) for x in range(6))
    window = MASWWindow(
        xmid=2.5,
        selected_files=[Path("a.dat"), Path("b.dat")],
        receiver_indices=list(range(6)),
        acquisitions=[
            LinearAcquisition(source=Coordinate(-3.0, 0.0, 0.0), receivers=receivers),
            LinearAcquisition(source=Coordinate(6.5, 0.0, 0.0), receivers=receivers),
        ],
    )

    # 3 m before the first receiver, 1.5 m past the last.
    assert nearest_offset(window) == 1.5
    assert nearest_offset(window.model_copy(update={"acquisitions": []})) is None
