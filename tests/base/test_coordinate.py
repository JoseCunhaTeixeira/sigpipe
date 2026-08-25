import math

import pytest

from sigpipe.base.coordinate import (
    UNKNOWN_COORDINATE,
    Coordinate,
    coordinates_to_tuples,
    tuples_to_coordinates,
)


def test_post_init_coerces_to_float() -> None:
    c = Coordinate(1, 2, 3)  # type: ignore[arg-type]
    assert c.x == 1.0
    assert isinstance(c.x, float)


def test_from_tuple_and_to_tuple_roundtrip() -> None:
    t = (1.0, 2.0, 3.0)
    c = Coordinate.from_tuple(t)
    assert c.to_tuple() == t


def test_add_sub() -> None:
    a = Coordinate(1.0, 2.0, 3.0)
    b = Coordinate(4.0, 5.0, 6.0)
    assert a + b == Coordinate(5.0, 7.0, 9.0)
    assert b - a == Coordinate(3.0, 3.0, 3.0)


def test_mul_and_rmul() -> None:
    a = Coordinate(1.0, 2.0, 3.0)
    assert a * 2 == Coordinate(2.0, 4.0, 6.0)
    assert 2 * a == Coordinate(2.0, 4.0, 6.0)


def test_dot() -> None:
    a = Coordinate(1.0, 2.0, 3.0)
    b = Coordinate(4.0, 5.0, 6.0)
    assert a.dot(b) == 32.0


def test_cross() -> None:
    x = Coordinate(1.0, 0.0, 0.0)
    y = Coordinate(0.0, 1.0, 0.0)
    assert x.cross(y) == Coordinate(0.0, 0.0, 1.0)


def test_norm() -> None:
    a = Coordinate(3.0, 4.0, 0.0)
    assert a.norm() == pytest.approx(5.0)


def test_distance_to() -> None:
    a = Coordinate(0.0, 0.0, 0.0)
    b = Coordinate(3.0, 4.0, 0.0)
    assert a.distance_to(b) == pytest.approx(5.0)


def test_is_unknown() -> None:
    assert UNKNOWN_COORDINATE.is_unknown
    assert not Coordinate(0.0, 0.0, 0.0).is_unknown
    assert Coordinate(math.nan, 0.0, 0.0).is_unknown


def test_str() -> None:
    c = Coordinate(1.0, 2.0, 3.0)
    assert str(c) == "Coordinate[1.000;2.000;3.000]"


def test_coordinates_to_tuples_and_back() -> None:
    coords = (Coordinate(1.0, 2.0, 3.0), Coordinate(4.0, 5.0, 6.0))
    tuples = coordinates_to_tuples(coords)
    assert tuples == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert tuples_to_coordinates(tuples) == coords


def test_frozen_dataclass_is_immutable() -> None:
    c = Coordinate(1.0, 2.0, 3.0)
    with pytest.raises(AttributeError):
        c.x = 5.0  # type: ignore[misc]
