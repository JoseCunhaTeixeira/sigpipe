import math
from collections.abc import Sequence
from dataclasses import dataclass

type TupleCoordinate = tuple[float, float, float]


@dataclass(slots=True, frozen=True)
class Coordinate:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "z", float(self.z))

    @classmethod
    def from_tuple(cls, t: TupleCoordinate) -> Coordinate:
        x, y, z = t
        return cls(float(x), float(y), float(z))

    def to_tuple(self) -> TupleCoordinate:
        return self.x, self.y, self.z

    def __add__(self, other: Coordinate) -> Coordinate:
        return Coordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Coordinate) -> Coordinate:
        return Coordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Coordinate:
        return Coordinate(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Coordinate:
        return self.__mul__(scalar)

    def dot(self, other: Coordinate) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Coordinate) -> Coordinate:
        return Coordinate(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def distance_to(self, other: Coordinate) -> float:
        return (self - other).norm()

    @property
    def is_unknown(self) -> bool:
        return math.isnan(self.x) or math.isnan(self.y) or math.isnan(self.z)

    def __str__(self) -> str:
        return f"Coordinate[{self.x:.3f};{self.y:.3f};{self.z:.3f}]"


def coordinates_to_tuples(
    coordinates: Sequence[Coordinate],
) -> tuple[TupleCoordinate, ...]:
    return tuple(p.to_tuple() for p in coordinates)


def tuples_to_coordinates(
    coords: Sequence[TupleCoordinate],
) -> tuple[Coordinate, ...]:
    return tuple(Coordinate.from_tuple(t) for t in coords)


UNKNOWN_COORDINATE = Coordinate(float("nan"), float("nan"), float("nan"))
