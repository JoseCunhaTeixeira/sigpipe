from collections.abc import Sequence
from pathlib import Path

import pytest

from sigpipe.base.pipeline import Pipeline
from sigpipe.base.transformer import Transformer


class Increment(Transformer[int, int]):
    def transform(self, data: Sequence[int]) -> list[int]:
        return [x + 1 for x in data]


class Double(Transformer[int, int]):
    def transform(self, data: Sequence[int]) -> list[int]:
        return [x * 2 for x in data]


class Explode(Transformer[int, int]):
    def transform(self, _data: Sequence[int]) -> list[int]:
        raise RuntimeError("boom")


def test_run_applies_steps_in_order() -> None:
    pipeline = Pipeline([Increment(), Double()])
    result = pipeline.run([1, 2, 3], show_log=False)
    assert result == [4, 6, 8]


def test_rshift_with_transformer_appends_step() -> None:
    pipeline = Pipeline([Increment()])
    combined = pipeline >> Double()
    assert [type(s) for s in combined.steps] == [Increment, Double]
    assert [type(s) for s in pipeline.steps] == [Increment]


def test_rshift_with_pipeline_concatenates_steps() -> None:
    p1 = Pipeline([Increment()])
    p2 = Pipeline([Double()])
    combined = p1 >> p2
    assert [type(s) for s in combined.steps] == [Increment, Double]


def test_repr_joins_step_names() -> None:
    pipeline = Pipeline([Increment(), Double()])
    assert repr(pipeline) == "Increment >> Double"


def test_run_propagates_step_exceptions() -> None:
    pipeline = Pipeline([Increment(), Explode()])
    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run([1], show_log=False)


def test_run_with_no_steps_returns_input_unchanged() -> None:
    pipeline = Pipeline([])
    assert pipeline.run("anything", show_log=False) == "anything"


def test_run_creates_log_file_when_save_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline([Increment()])
    pipeline.run([1], save_log=True, show_log=True)

    log_dir = tmp_path / "logs"
    assert log_dir.is_dir()
    assert list(log_dir.glob("*.log"))
