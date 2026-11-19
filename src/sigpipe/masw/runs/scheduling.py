"""Work in parts that a pool's workers share: each part runs on the first worker free, then the
work's last step on the parts' outputs, in a worker too. A line's windows inverted so keep every
worker busy while chains remain: each window's chains on whichever worker is free, the costliest
first, then its files as soon as its chains are in.

The thread that submitted the work drives the pool (Scheduler.finished): it hands the pool no
more steps than it has workers, so that a work's last step starts as soon as its parts are in,
ahead of the parts still waiting, and a stop finds nothing queued in the pool. A stop kills the
workers at once, as sigpipe.masw.runs.stopping.finished does."""

from __future__ import annotations

import heapq
import itertools
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from sigpipe.masw.runs.stopping import POLL_S, Stopped


@dataclass(frozen=True, slots=True)
class Split[T]:
    """Work in two steps: its `parts`, each run on its own in any worker, called without
    arguments, then `finish` on their outputs, in the parts' order, in a worker too (both pickled
    to the workers). `cost`: a part's expected time, relative to other work's: the costliest
    parts start first."""

    parts: Sequence[Callable[[], Any]]
    finish: Callable[[list[Any]], T]
    cost: float = 1.0


@dataclass(slots=True, eq=False)
class _Work:
    """A split submitted: its future, its parts' outputs, the parts not in yet."""

    split: Split[Any]
    future: Future[Any]
    outputs: list[Any]
    left: int


class Scheduler:
    """Splits run on a pool of `workers` processes, each started with `initializer(*initargs)`.
    A context manager: the pool is shut down on leaving it, its workers killed at once when an
    error leaves it."""

    def __init__(
        self,
        workers: int,
        initializer: Callable[..., object] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        self._workers = max(1, workers)
        self._executor = ProcessPoolExecutor(
            max_workers=self._workers, initializer=initializer, initargs=initargs
        )
        # The steps waiting, as (0 for a last step or 1 for a part, the part's cost negated, the
        # order submitted, its work, the part's index or None for the last step): a heap.
        self._queue: list[tuple[int, float, int, _Work, int | None]] = []
        self._running: dict[Future[Any], tuple[_Work, int | None]] = {}
        self._order = itertools.count()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._kill()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def submit[T](self, split: Split[T]) -> Future[T]:
        """`split` to run: its future holds its last step's result once the loop (finished) ran
        it, or the first error of its steps."""
        future: Future[T] = Future()
        work = _Work(split, future, [None] * len(split.parts), len(split.parts))
        steps = range(len(split.parts)) if split.parts else (None,)  # no parts: its last step
        for index in steps:
            self._push(work, index)
        return future

    def finished[T](
        self, futures: Iterable[Future[T]], stop: threading.Event | None = None
    ) -> Iterator[Future[T]]:
        """`futures` as their work finishes, the pool fed meanwhile: submit's, and any already
        done (a work refused before it was submitted, its error set). On a stop: the workers
        killed at once, the steps waiting dropped, the work that had just finished yielded, and
        Stopped raised."""
        pending = set(futures)
        while pending:
            if stop is not None and stop.is_set():
                self._kill()
                yield from (
                    future
                    for future in pending
                    if future.done() and not future.cancelled() and future.exception() is None
                )
                raise Stopped()
            self._start()
            done = {future for future in pending if future.done()}
            if done:
                pending -= done
                yield from done
                continue
            if not self._running:
                raise RuntimeError("Futures not submitted to this scheduler")
            ready, _ = wait(
                self._running,
                timeout=POLL_S if stop is not None else None,
                return_when=FIRST_COMPLETED,
            )
            for step in ready:
                self._settle(step)

    def _push(self, work: _Work, index: int | None) -> None:
        last = 0 if index is None else 1
        heapq.heappush(self._queue, (last, -work.split.cost, next(self._order), work, index))

    def _start(self) -> None:
        """The workers free given the steps waiting: the last steps first, then the costliest
        parts. The steps of a work that failed or was cancelled are dropped."""
        while self._queue and len(self._running) < self._workers:
            _, _, _, work, index = heapq.heappop(self._queue)
            if work.future.done():
                continue
            try:
                step = (
                    self._executor.submit(work.split.finish, work.outputs)
                    if index is None
                    else self._executor.submit(work.split.parts[index])
                )
            except Exception as error:  # a broken pool: a worker died
                work.future.set_exception(error)
                continue
            self._running[step] = (work, index)

    def _settle(self, step: Future[Any]) -> None:
        """A step that ended: its part's output kept (its work's last step queued once every
        part is in), or its work's result, or its error the work's."""
        work, index = self._running.pop(step)
        if work.future.done():
            return
        error = CancelledError() if step.cancelled() else step.exception()
        if error is not None:
            work.future.set_exception(error)
        elif index is None:
            work.future.set_result(step.result())
        else:
            work.outputs[index] = step.result()
            work.left -= 1
            if work.left == 0:
                self._push(work, None)

    def _kill(self) -> None:
        """The workers killed at once, the steps waiting dropped, the work not finished
        cancelled."""
        self._executor.kill_workers()
        works = [work for *_, work, _ in self._queue] + [work for work, _ in self._running.values()]
        self._queue.clear()
        self._running.clear()
        for work in works:
            work.future.cancel()
