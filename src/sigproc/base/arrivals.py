from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Arrival:
    label: str
    time: float
    amplitude: float
    residual_phase: float | None = None


@dataclass(frozen=True)
class TraceArrivals:
    arrivals: tuple[Arrival, ...] = ()

    def get(self, label: str) -> Arrival | None:
        for arrival in self.arrivals:
            if arrival.label == label:
                return arrival
        return None

    def __iter__(self) -> Iterator[Arrival]:
        return iter(self.arrivals)

    def __len__(self) -> int:
        return len(self.arrivals)

    def __getitem__(self, item: int) -> Arrival:
        return self.arrivals[item]
