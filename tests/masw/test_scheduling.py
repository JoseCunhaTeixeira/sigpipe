"""Work in parts on a pool's workers (sigpipe.masw.runs.scheduling): every work's parts at once,
the costliest first, each work's last step as soon as its parts are in, a part's failure its
work's alone, and a stop that kills the workers at once."""

import os
import threading
import time
from concurrent.futures import Future
from functools import partial
from pathlib import Path

import pytest

from sigpipe.masw.runs.scheduling import Scheduler, Split
from sigpipe.masw.runs.stopping import Stopped


def _together(folder: Path, name: str, expected: int) -> str:
    """A part that waits in `folder` until `expected` parts have arrived there: they all run at
    once, or it times out."""
    (folder / name).write_text(str(os.getpid()), encoding="utf-8")
    deadline = time.monotonic() + 120
    while len(list(folder.iterdir())) < expected:
        if time.monotonic() > deadline:
            raise TimeoutError(f"{name}: the parts did not all run at once")
        time.sleep(0.05)
    return name


def _logged(log: Path, name: str, fail: bool = False) -> str:
    """A part that writes its name in `log` as it starts (one worker: one after the other)."""
    with log.open("a", encoding="utf-8") as file:
        file.write(name + "\n")
    if fail:
        raise ValueError(f"{name} failed")
    return name


def _joined(log: Path, name: str, outputs: list[str]) -> str:
    """A last step that writes its name in `log`, then joins its parts' outputs."""
    _logged(log, name)
    return "+".join(outputs)


def test_every_works_parts_share_the_workers(tmp_path: Path) -> None:
    # Two windows of three chains on six workers: the six at once, in six processes. Their
    # outputs reach each window's last step in the parts' order.
    log = tmp_path / "log.txt"
    arrivals = tmp_path / "arrivals"
    arrivals.mkdir()
    with Scheduler(6) as scheduler:
        futures = {
            scheduler.submit(
                Split(
                    parts=[partial(_together, arrivals, f"{work}{i}", 6) for i in range(3)],
                    finish=partial(_joined, log, f"{work} done"),
                )
            ): work
            for work in "AB"
        }
        results = {futures[future]: future.result() for future in scheduler.finished(futures)}

    assert results == {"A": "A0+A1+A2", "B": "B0+B1+B2"}
    assert len({path.read_text(encoding="utf-8") for path in arrivals.iterdir()}) == 6


def test_the_costliest_parts_start_first_and_a_last_step_once_its_parts_are_in(
    tmp_path: Path,
) -> None:
    log = tmp_path / "log.txt"
    with Scheduler(1) as scheduler:
        cheap = scheduler.submit(
            Split(
                [partial(_logged, log, f"cheap{i}") for i in range(2)],
                partial(_joined, log, "cheap done"),
                cost=1.0,
            )
        )
        costly = scheduler.submit(
            Split(
                [partial(_logged, log, f"costly{i}") for i in range(2)],
                partial(_joined, log, "costly done"),
                cost=5.0,
            )
        )
        # A work of no parts: its last step alone.
        alone = scheduler.submit(Split((), partial(_joined, log, "alone"), cost=0.0))
        done = list(scheduler.finished([cheap, costly, alone]))

    # The last steps ahead of the parts waiting: the work of no parts at once, the costly work's
    # as soon as its parts are in.
    assert log.read_text(encoding="utf-8").splitlines() == [
        "alone",
        "costly0",
        "costly1",
        "costly done",
        "cheap0",
        "cheap1",
        "cheap done",
    ]
    assert done == [alone, costly, cheap]
    assert (costly.result(), alone.result(), cheap.result()) == (
        "costly0+costly1",
        "",
        "cheap0+cheap1",
    )


def test_a_part_that_fails_fails_its_work_alone(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    with Scheduler(1) as scheduler:
        failing = scheduler.submit(
            Split(
                [
                    partial(_logged, log, "a0"),
                    partial(_logged, log, "a1", fail=True),
                    partial(_logged, log, "a2"),
                ],
                partial(_joined, log, "a done"),
                cost=2.0,
            )
        )
        other = scheduler.submit(
            Split([partial(_logged, log, "b0")], partial(_joined, log, "b done"))
        )
        # A work refused before it was submitted: its error its outcome, at once.
        refused: Future[str] = Future()
        refused.set_exception(ValueError("refused"))
        done = list(scheduler.finished([failing, other, refused]))

    assert done[0] is refused and set(done) == {failing, other, refused}
    with pytest.raises(ValueError, match="a1 failed"):
        failing.result()
    assert other.result() == "b0"
    # The failed work's part still waiting dropped, its last step never run.
    assert log.read_text(encoding="utf-8").splitlines() == ["a0", "a1", "b0", "b done"]


def test_a_stop_kills_the_workers_at_once_and_keeps_the_work_finished(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    stop = threading.Event()
    kept: list[Future[str]] = []
    start = time.monotonic()

    with pytest.raises(Stopped), Scheduler(2) as scheduler:
        quick = scheduler.submit(Split([partial(_logged, log, "quick")], partial(_joined, log, "")))
        slow = scheduler.submit(Split([partial(time.sleep, 600)], partial(_joined, log, "slow")))
        for future in scheduler.finished([quick, slow], stop):
            kept.append(future)
            stop.set()  # stopped as the quick work finishes

    assert kept == [quick] and quick.result() == "quick"
    assert slow.cancelled()
    assert time.monotonic() - start < 60  # not the 600 s the slow part sleeps
