"""Writing a run safely: a file replaced whole or not at all, and one application writing a run
at a time (PACo exclusively, PAC's pages shared among them)."""

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from sigpipe.masw.runs.writing import RunBusy, run_lock, write_atomic


def _in_thread(work: Callable[[], object]) -> BaseException | None:
    """What `work` raised in another thread (another writer), or None."""
    raised: list[BaseException] = []

    def run() -> None:
        try:
            work()
        except BaseException as error:
            raised.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=30)
    return raised[0] if raised else None


def _hold(run: Path, owner: str, shared: bool, wait_s: float = 0.0) -> Callable[[], None]:
    def hold() -> None:
        with run_lock(run, owner, shared=shared, wait_s=wait_s):
            pass

    return hold


def test_a_file_is_replaced_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run.json"
    write_atomic(path, "old")
    write_atomic(path, "new")
    assert path.read_text() == "new"

    def full_disk(*_: object) -> Path:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", full_disk)
    with pytest.raises(OSError, match="No space left"):
        write_atomic(path, "newer")
    # The old file whole, nothing half-written beside it.
    assert path.read_text() == "new" and sorted(tmp_path.iterdir()) == [path]


def test_one_writer_holds_a_run_exclusively(tmp_path: Path) -> None:
    with run_lock(tmp_path, "PACo: pick"):
        refused = _in_thread(_hold(tmp_path, "PAC", shared=True))
        other = _in_thread(_hold(tmp_path, "PACo: invert", shared=False))
        # The same thread takes it again freely: a tool calling another.
        with run_lock(tmp_path, "PACo: judge"):
            pass

    assert isinstance(refused, RunBusy) and isinstance(other, RunBusy)
    assert str(refused) == (
        f"Run {tmp_path.name} is being written by PACo: pick: try again when it ends."
    )
    # Released: free again, the holder's name gone.
    assert _in_thread(_hold(tmp_path, "PAC", shared=True)) is None
    assert (tmp_path / ".run.lock").read_text() == ""


def test_pacs_pages_share_a_run_and_keep_paco_out(tmp_path: Path) -> None:
    with run_lock(tmp_path, "PAC", shared=True):
        assert _in_thread(_hold(tmp_path, "PAC", shared=True)) is None
        refused = _in_thread(_hold(tmp_path, "PACo: pick", shared=False))

    assert isinstance(refused, RunBusy)
    assert "being written by one of PAC's pages" in str(refused)


def test_a_writer_waits_for_the_run_up_to_its_limit(tmp_path: Path) -> None:
    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with run_lock(tmp_path, "PAC", shared=True):
            held.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(timeout=10)
    threading.Timer(0.3, release.set).start()

    # Released within the wait: taken.
    assert _in_thread(_hold(tmp_path, "PACo: pick", shared=False, wait_s=5)) is None
    thread.join(timeout=10)
