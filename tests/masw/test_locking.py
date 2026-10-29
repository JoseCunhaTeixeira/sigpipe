"""Locks on files, alike on every system: each open file holds its own, shared or exclusive; a
locked file stays readable and writable to the others."""

import threading
import time
from pathlib import Path

import pytest

from sigpipe.masw.runs.locking import lock_file, unlock_file


def test_an_exclusive_lock_refuses_any_other(tmp_path: Path) -> None:
    path = tmp_path / ".run.lock"
    with path.open("a+") as held, path.open("a+") as other:
        lock_file(held)

        with pytest.raises(BlockingIOError):
            lock_file(other, wait=False)
        with pytest.raises(BlockingIOError):
            lock_file(other, shared=True, wait=False)

        unlock_file(held)
        lock_file(other, wait=False)
        unlock_file(other)


def test_shared_locks_share_and_refuse_an_exclusive_one(tmp_path: Path) -> None:
    path = tmp_path / ".run.lock"
    with path.open("a+") as first, path.open("a+") as second, path.open("a+") as writer:
        lock_file(first, shared=True)
        lock_file(second, shared=True, wait=False)

        with pytest.raises(BlockingIOError):
            lock_file(writer, wait=False)

        unlock_file(first)
        unlock_file(second)
        lock_file(writer, wait=False)
        unlock_file(writer)


def test_a_locked_file_stays_readable_and_writable_to_others(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    with path.open("a+") as held:
        lock_file(held)
        held.write("PACo: pick\n")
        held.flush()

        # The holder's name read by a writer it refuses; a line added beside it.
        assert path.read_text() == "PACo: pick\n"
        with path.open("a") as other:
            other.write("PAC\n")

        unlock_file(held)
    assert path.read_text() == "PACo: pick\nPAC\n"


def test_a_lock_waited_for_is_taken_once_released(tmp_path: Path) -> None:
    path = tmp_path / ".qc_log.lock"
    held, released = threading.Event(), threading.Event()

    def holder() -> None:
        with path.open("a") as lock:
            lock_file(lock)
            held.set()
            time.sleep(0.3)
            released.set()
            unlock_file(lock)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(timeout=10)
    with path.open("a") as lock:
        lock_file(lock)
        assert released.is_set()
        unlock_file(lock)
    thread.join(timeout=10)
