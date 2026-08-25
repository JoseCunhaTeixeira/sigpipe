from collections.abc import Sequence

import pytest

from sigpipe.base.pipeline import Pipeline
from sigpipe.base.transformer import Transformer


class Increment(Transformer[int, int]):
    def transform(self, data: Sequence[int]) -> list[int]:
        return [x + 1 for x in data]


def test_rshift_between_transformers_builds_pipeline() -> None:
    a, b = Increment(), Increment()
    pipeline = a >> b
    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps == [a, b]


def test_rshift_transformer_and_pipeline() -> None:
    a, b, c = Increment(), Increment(), Increment()
    pipeline = a >> (b >> c)
    assert pipeline.steps == [a, b, c]


def test_name_defaults_to_class_name() -> None:
    assert Increment().name == "Increment"


@pytest.mark.parametrize("data", [None, 5, "abcd", b"abcd"])
def test_validate_sequence_rejects_non_sequence(data: object) -> None:
    with pytest.raises(TypeError, match="Expected Sequence"):
        Transformer.validate_sequence(data)


def test_validate_sequence_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Empty input sequence"):
        Transformer.validate_sequence([])


def test_validate_sequence_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="All elements must be int"):
        Transformer.validate_sequence([1, "two", 3], int)


def test_validate_sequence_accepts_multiple_expected_types() -> None:
    Transformer.validate_sequence([1, 2.0, 3], int, float)


def test_validate_sequence_no_type_check_when_no_expected_types() -> None:
    Transformer.validate_sequence([1, "two", object()])


def test_validate_homogeneous_sequence_rejects_mixed_types() -> None:
    with pytest.raises(TypeError, match="same type"):
        Transformer.validate_homogeneous_sequence([1, "two"])


def test_validate_homogeneous_sequence_accepts_same_type() -> None:
    Transformer.validate_homogeneous_sequence([1, 2, 3])


def test_validate_homogeneous_sequence_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Empty input sequence"):
        Transformer.validate_homogeneous_sequence([])
