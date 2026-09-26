"""Building the MASW windows of a profile."""

import logging
from pathlib import Path
from typing import Literal

from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.masw.profiles import Profile
from sigpipe.masw.windows.models import Exclusions, MASWParameters, MASWWindow

logger = logging.getLogger(__name__)


def build_windows(profile: Profile, params: MASWParameters) -> list[MASWWindow]:
    n_receivers = len(profile.receivers)
    if params.length > n_receivers:
        raise ValueError(
            f"length ({params.length}) exceeds the {n_receivers} receivers "
            f"of profile '{profile.name}'."
        )

    windows: list[MASWWindow] = []

    for start in range(0, n_receivers - params.length + 1, params.step):
        stop = start + params.length
        receivers = profile.receivers[start:stop]
        xmin, xmax = receivers[0].x, receivers[-1].x

        # sigpipe finds the window middle by arc length along (x, z). The source plays no part
        # in it, so the first receiver fills that slot.
        xmid = LinearAcquisition(source=receivers[0], receivers=receivers).xmid

        selected_files: list[Path] = []
        acquisitions: list[LinearAcquisition] = []

        for record in profile.records:
            source: Coordinate
            if record.source is None:
                # Passive: every record serves every window. LinearAcquisition needs a known
                # source, so the first receiver stands in for it; cross-correlation redefines
                # the source later anyway.
                source = receivers[0]
            else:
                source = record.source

                # The source must be outside the window...
                if xmin < source.x < xmax:
                    continue

                # ...and at a usable distance from its middle.
                distance = abs(source.x - xmid)
                if distance <= params.distance_min or distance >= params.distance_max:
                    continue

            selected_files.append(record.path)
            acquisitions.append(LinearAcquisition(source=source, receivers=receivers))

        if not selected_files:
            logger.warning("No valid shots for xmid=%.2f", xmid)
            continue

        windows.append(
            MASWWindow(
                xmid=xmid,
                selected_files=selected_files,
                receiver_indices=list(range(start, stop)),
                acquisitions=acquisitions,
            )
        )

    logger.info("Built %d valid MASW windows", len(windows))

    return windows


# How a window's records share its receivers once a run's exclusions are applied: each its own
# (active, shots imaged one by one), one set with the records bringing other exclusions left out
# (passive-active, correlation gathers stacked sample by sample), or the union (passive).
type Geometry = Literal["per_record", "shared", "union"]


def apply_exclusions(
    window: MASWWindow, exclusions: Exclusions, geometry: Geometry = "union"
) -> MASWWindow | None:
    """`window` without the records and traces excluded; None when fewer than 3 receivers or
    no record are left. The xmid stays: an exclusion never moves a window. "per_record" (active
    windows, whose shots are imaged one by one): each record leaves out only its own excluded
    traces, and a record left with fewer than 3 receivers leaves the window. "shared"
    (passive-active windows, whose correlation gathers are stacked): the receivers at least half
    the records excluded leave every record, and a record that excluded any other receiver of
    the window leaves it. "union" (passive windows, whose records are correlated trace by
    trace): a trace any record excluded leaves every record. With many shots a window, the union
    can leave no receiver."""
    if geometry == "per_record":
        return _per_record(window, exclusions)
    if geometry == "shared":
        return _shared(window, exclusions)
    kept = [
        index
        for index, path in enumerate(window.selected_files)
        if path.name not in exclusions.records
    ]
    files = [window.selected_files[index] for index in kept]
    dropped = {trace for path in files for trace in exclusions.traces.get(path.name, ())}
    positions = [
        position
        for position, receiver in enumerate(window.receiver_indices)
        if receiver not in dropped
    ]
    if not files or len(positions) < 3:
        return None
    acquisitions = [
        LinearAcquisition(
            source=window.acquisitions[index].source,
            receivers=tuple(
                window.acquisitions[index].receivers[position] for position in positions
            ),
        )
        for index in kept
    ]
    return MASWWindow(
        xmid=window.xmid,
        selected_files=files,
        receiver_indices=[window.receiver_indices[position] for position in positions],
        acquisitions=acquisitions,
    )


def _shared(window: MASWWindow, exclusions: Exclusions) -> MASWWindow | None:
    """`window` with one set of receivers for every record it keeps: the receivers at least half
    its records excluded leave, and so does every record that excluded another of its
    receivers."""
    records = [
        (path, acquisition)
        for path, acquisition in zip(window.selected_files, window.acquisitions, strict=True)
        if path.name not in exclusions.records
    ]
    inside = set(window.receiver_indices)
    counts: dict[int, int] = {}
    for path, _ in records:
        for trace in set(exclusions.traces.get(path.name, ())) & inside:
            counts[trace] = counts.get(trace, 0) + 1
    shared = {trace for trace, count in counts.items() if 2 * count >= len(records)}
    kept = [
        (path, acquisition)
        for path, acquisition in records
        if set(exclusions.traces.get(path.name, ())) & inside <= shared
    ]
    positions = [
        position
        for position, receiver in enumerate(window.receiver_indices)
        if receiver not in shared
    ]
    if not kept or len(positions) < 3:
        return None
    return MASWWindow(
        xmid=window.xmid,
        selected_files=[path for path, _ in kept],
        receiver_indices=[window.receiver_indices[position] for position in positions],
        acquisitions=[
            LinearAcquisition(
                source=acquisition.source,
                receivers=tuple(acquisition.receivers[position] for position in positions),
            )
            for _, acquisition in kept
        ],
    )


def _per_record(window: MASWWindow, exclusions: Exclusions) -> MASWWindow | None:
    """`window` with each record giving the receivers it did not exclude, and none giving a
    receiver at least half the window's records excluded: a receiver's defect (a dead geophone,
    bad coupling) shows in most of its shots, while one record's (a clipped trace, a near-field
    amplitude) is its own. With two records this is the union."""
    records = [
        (path, acquisition)
        for path, acquisition in zip(window.selected_files, window.acquisitions, strict=True)
        if path.name not in exclusions.records
    ]
    counts: dict[int, int] = {}
    for path, _ in records:
        for trace in set(exclusions.traces.get(path.name, ())):
            counts[trace] = counts.get(trace, 0) + 1
    shared = {trace for trace, count in counts.items() if 2 * count >= len(records)}
    files: list[Path] = []
    acquisitions: list[LinearAcquisition] = []
    own: list[list[int]] = []
    for path, acquisition in records:
        dropped = set(exclusions.traces.get(path.name, ())) | shared
        positions = [
            position
            for position, receiver in enumerate(window.receiver_indices)
            if receiver not in dropped
        ]
        if len(positions) < 3:
            continue
        files.append(path)
        own.append([window.receiver_indices[position] for position in positions])
        acquisitions.append(
            LinearAcquisition(
                source=acquisition.source,
                receivers=tuple(acquisition.receivers[position] for position in positions),
            )
        )
    if not files:
        return None
    untouched = all(len(indices) == len(window.receiver_indices) for indices in own)
    return MASWWindow(
        xmid=window.xmid,
        selected_files=files,
        receiver_indices=window.receiver_indices,
        acquisitions=acquisitions,
        record_receivers=None if untouched else own,
    )
