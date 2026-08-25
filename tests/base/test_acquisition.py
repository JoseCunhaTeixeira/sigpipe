import numpy as np
import pytest

from sigpipe.base.acquisition import (
    Acquisition,
    LinearAcquisition,
    PlanarAcquisition,
    acquisition_from_kind,
    acquisition_kind,
)
from sigpipe.base.coordinate import UNKNOWN_COORDINATE, Coordinate


def test_base_offsets_and_mid_position_not_implemented() -> None:
    acquisition = Acquisition(
        source=UNKNOWN_COORDINATE,
        receivers=(UNKNOWN_COORDINATE,),
    )
    with pytest.raises(NotImplementedError):
        _ = acquisition.offsets
    with pytest.raises(NotImplementedError):
        _ = acquisition.mid_position


def test_planar_offsets_are_straight_chords() -> None:
    acquisition = PlanarAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(3.0, 4.0, 0.0), Coordinate(0.0, 5.0, 0.0)),
    )
    np.testing.assert_allclose(acquisition.offsets, [5.0, 5.0])


def test_planar_mid_position_is_centroid() -> None:
    acquisition = PlanarAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(2.0, 0.0, 0.0),),
    )
    mid = acquisition.mid_position
    assert mid.x == pytest.approx(1.0)


def test_planar_requires_known_positions() -> None:
    with pytest.raises(ValueError, match="known positions"):
        PlanarAcquisition(
            source=UNKNOWN_COORDINATE,
            receivers=(Coordinate(1.0, 0.0, 0.0),),
        )


def test_planar_requires_equal_z() -> None:
    with pytest.raises(ValueError, match="z coordinates"):
        PlanarAcquisition(
            source=Coordinate(0.0, 0.0, 0.0),
            receivers=(Coordinate(1.0, 0.0, 1.0),),
        )


def test_linear_offsets_follow_flat_profile() -> None:
    acquisition = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(1.0, 0.0, 0.0), Coordinate(2.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose(acquisition.offsets, [1.0, 2.0])


def test_linear_offsets_follow_topographic_profile() -> None:
    acquisition = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(3.0, 0.0, 4.0),),
    )
    np.testing.assert_allclose(acquisition.offsets, [5.0])


def test_linear_requires_known_positions() -> None:
    with pytest.raises(ValueError, match="known positions"):
        LinearAcquisition(
            source=UNKNOWN_COORDINATE,
            receivers=(Coordinate(1.0, 0.0, 0.0),),
        )


def test_linear_requires_equal_y() -> None:
    with pytest.raises(ValueError, match="y coordinates"):
        LinearAcquisition(
            source=Coordinate(0.0, 0.0, 0.0),
            receivers=(Coordinate(1.0, 1.0, 0.0),),
        )


def test_linear_mid_position_single_receiver() -> None:
    receiver = Coordinate(1.0, 0.0, 0.0)
    acquisition = LinearAcquisition(source=Coordinate(0.0, 0.0, 0.0), receivers=(receiver,))
    assert acquisition.mid_position == receiver


def test_linear_mid_position_splits_line_in_half() -> None:
    acquisition = LinearAcquisition(
        source=Coordinate(-1.0, 0.0, 0.0),
        receivers=(
            Coordinate(0.0, 0.0, 0.0),
            Coordinate(10.0, 0.0, 0.0),
        ),
    )
    mid = acquisition.mid_position
    assert mid.x == pytest.approx(5.0)
    assert mid.z == pytest.approx(0.0)


def test_xmid_matches_mid_position_x() -> None:
    acquisition = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(0.0, 0.0, 0.0), Coordinate(4.0, 0.0, 0.0)),
    )
    assert acquisition.xmid == pytest.approx(acquisition.mid_position.x)


def test_is_unknown() -> None:
    known = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(1.0, 0.0, 0.0),),
    )
    assert not known.is_unknown

    unknown = Acquisition(source=UNKNOWN_COORDINATE, receivers=(UNKNOWN_COORDINATE,))
    assert unknown.is_unknown


def test_equality_compares_source_and_receivers() -> None:
    a = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(1.0, 0.0, 0.0),),
    )
    b = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(1.0, 0.0, 0.0),),
    )
    c = LinearAcquisition(
        source=Coordinate(0.0, 0.0, 0.0),
        receivers=(Coordinate(2.0, 0.0, 0.0),),
    )
    assert a == b
    assert a != c
    assert a != "not an acquisition"


@pytest.mark.parametrize(
    ("cls", "kind"),
    [
        (Acquisition, "Acquisition"),
        (PlanarAcquisition, "PlanarAcquisition"),
        (LinearAcquisition, "LinearAcquisition"),
    ],
)
def test_acquisition_kind_roundtrip(cls: type[Acquisition], kind: str) -> None:
    source = Coordinate(0.0, 0.0, 0.0)
    receivers = (Coordinate(1.0, 0.0, 0.0),)
    acquisition = cls(source=source, receivers=receivers)

    assert acquisition_kind(acquisition) == kind

    rebuilt = acquisition_from_kind(kind, source, receivers)
    assert type(rebuilt) is cls
    assert rebuilt == acquisition


def test_acquisition_from_kind_falls_back_to_base_for_unknown_kind() -> None:
    source = Coordinate(0.0, 0.0, 0.0)
    receivers = (Coordinate(1.0, 0.0, 0.0),)
    rebuilt = acquisition_from_kind("SomeFutureKind", source, receivers)
    assert type(rebuilt) is Acquisition
