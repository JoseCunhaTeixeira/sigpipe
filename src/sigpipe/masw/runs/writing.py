"""Writing a run safely (S5 of PACo's agent guidelines): a file replaced whole or not at all, and
one application writing a run at a time.

The run's lock is a lock file in its folder, `.run.lock`. PACo's tools hold it exclusively,
one writer at all; PAC's pages hold it shared, so that they do not exclude each other (a
curve picked while PAC inverts the run) but exclude PACo, and PACo excludes them. A writer
that finds the run held waits up to `wait_s`, then is refused with who holds it. A thread
holding the run takes it again freely (a tool calling another that writes the same run)."""

import fcntl
import os
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from sigpipe.masw.runs.models import RunError

RUN_LOCK = ".run.lock"
# How often a writer waiting for the run tries again.
_POLL_S = 0.1


class RunBusy(RunError):
    """Another application, or another task, writes the run."""


def write_atomic(path: Path, text: str) -> None:
    """`text` written to `path` whole or not at all: a temporary file beside it, renamed over it
    (a crash leaves the old file or the new one, never half of one)."""
    partial = path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.partial")
    try:
        partial.write_text(text)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


_held = threading.local()


@contextmanager
def run_lock(
    run_folder: Path, owner: str, *, shared: bool = False, wait_s: float = 0.0
) -> Generator[None]:
    """Run `run_folder` held for `owner` ("PACo: pick") while the block writes it: exclusively,
    or `shared` with the other shared writers. Raises RunBusy, naming the exclusive holder, when
    the run stays held otherwise past `wait_s` seconds."""
    holding: dict[Path, int] = getattr(_held, "runs", None) or {}
    _held.runs = holding
    key = run_folder.resolve()
    if holding.get(key):
        holding[key] += 1
        try:
            yield
        finally:
            holding[key] -= 1
        return
    with (run_folder / RUN_LOCK).open("a+") as lock:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(lock, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    lock.seek(0)
                    holder = lock.read().strip() or "one of PAC's pages"
                    raise RunBusy(
                        f"Run {run_folder.name} is being written by {holder}: try again when "
                        "it ends."
                    ) from None
                time.sleep(_POLL_S)
        holding[key] = 1
        try:
            if not shared:
                _say(lock, owner)
            yield
        finally:
            holding.pop(key, None)
            if not shared:
                _say(lock, "")
            fcntl.flock(lock, fcntl.LOCK_UN)


def _say(lock: TextIO, owner: str) -> None:
    """The exclusive holder named in the lock file, for the writers it refuses."""
    lock.seek(0)
    lock.truncate()
    lock.write(owner)
    lock.flush()
