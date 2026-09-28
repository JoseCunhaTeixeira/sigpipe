import os

import pytest

from sigpipe.workers import THREAD_VARIABLES, one_thread_each


def test_the_processes_started_after_run_one_thread_each(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    # A setting of the user's stays.
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    one_thread_each()

    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert all(os.environ[name] == "1" for name in THREAD_VARIABLES if name != "OMP_NUM_THREADS")
