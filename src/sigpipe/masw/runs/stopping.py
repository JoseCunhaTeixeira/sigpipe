"""Stopping work cleanly, on request (a person's Stop button, or an agent's): the caller holds a
threading.Event and sets it. The loops that run tasks in worker processes check it while they
wait; on a stop they kill their workers at once, cancel the tasks queued, keep the tasks that
finished, undo the others and raise Stopped.

Nothing half-written is ever left: a task writes into a staging folder inside the folder its
outputs go to, and its outputs are moved into place only once it finished. Undoing a task removes
its staging folder, and its folder too when the task's call created it; outputs a task replaces
stay as they were until its own are complete."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

# A task's outputs while it runs, inside the folder they go to once it finished.
STAGING = ".partial"
# How often a loop waiting on its tasks looks at the stop, in seconds.
POLL_S = 0.2


class Stopped(BaseException):
    """The work stopped on request. `kept`: what the stopped call kept (a loop's finished
    outcomes, a run's manifest), or None when it kept nothing.

    A BaseException, as KeyboardInterrupt is: an `except Exception` around a task, which turns a
    failure into an outcome, must not swallow it."""

    def __init__(self, kept: object = None) -> None:
        super().__init__("Stopped on request.")
        self.kept = kept


def check(stop: threading.Event | None) -> None:
    """Raise Stopped when `stop` is set: between steps of work done in this process."""
    if stop is not None and stop.is_set():
        raise Stopped()


def finished[T](
    executor: ProcessPoolExecutor,
    futures: Iterable[Future[T]],
    stop: threading.Event | None,
    kill: bool = True,
) -> Iterator[Future[T]]:
    """`futures` as they finish. On a stop: the pool's queued tasks cancelled and its workers
    killed at once, the tasks that had just finished yielded, and Stopped raised; or, not to
    `kill` (short tasks that cannot be undone halfway), the running tasks yielded as they
    finish, then Stopped raised."""
    pending = set(futures)
    stopping = False
    while pending:
        if not stopping and stop is not None and stop.is_set():
            if kill:
                executor.kill_workers()
                for future in list(pending):
                    if future.done() and not future.cancelled() and future.exception() is None:
                        yield future
                raise Stopped()
            executor.shutdown(wait=False, cancel_futures=True)
            stopping = True
            pending = {future for future in pending if not future.cancelled()}
            continue
        done, pending = wait(
            pending, timeout=POLL_S if stop is not None else None, return_when=FIRST_COMPLETED
        )
        yield from done
    if stopping:
        raise Stopped()


def staging(folder: Path) -> Path:
    """A fresh staging folder in `folder` for a task about to run (one a killed process left,
    removed)."""
    path = folder / STAGING
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def commit(folder: Path) -> None:
    """A finished task's outputs moved from `folder`'s staging folder into `folder`, replacing
    those of the same names."""
    path = folder / STAGING
    if not path.exists():
        return
    for entry in path.iterdir():
        target = folder / entry.name
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() and entry.is_dir():
            target.unlink()
        entry.replace(target)
    path.rmdir()


def undo(folder: Path, created: bool) -> None:
    """A task that did not finish, undone: its staging folder removed, and `folder` too when the
    task's call `created` it."""
    shutil.rmtree(folder if created else folder / STAGING, ignore_errors=True)
