from dataclasses import dataclass
from typing import Literal

import numpy as np

from sigproc.base.coordinate import Coordinate


@dataclass(slots=True, frozen=True)
class VelocityModel1D:
    vs: tuple[float, ...]
    std: tuple[float, ...]
    thicknesses: tuple[float, ...]
    position: Coordinate
    label: Literal["Vp", "Vs", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        if not (len(self.thicknesses) == len(self.vs) == len(self.std)):
            raise ValueError(
                "thicknesses, vs, and std arrays must have the same length, "
                f"got {len(self.thicknesses)}, {len(self.vs)}, and {len(self.std)}"
            )

    @property
    def n_layers(self) -> int:
        return len(self.vs)

    @property
    def depths(self) -> tuple[float, ...]:
        """Depth of the bottom of each layer (cumulative thickness)."""
        return tuple(float(d) for d in np.cumsum(self.thicknesses))

    def _sample_values(
        self,
        elevations: np.ndarray,
        values: tuple[float, ...],
    ) -> np.ndarray:
        """
        Piecewise-constant sampling of layer values at given elevations.

        NaN only above the surface. The deepest layer extends downward
        indefinitely.
        """
        elevations = np.asarray(elevations, dtype=np.float32)
        depths = self.position.z - elevations

        bottoms = np.cumsum(np.asarray(self.thicknesses, dtype=np.float32))
        arr_values = np.asarray(values, dtype=np.float32)

        idx = np.searchsorted(bottoms, depths, side="right")
        idx = np.minimum(idx, len(arr_values) - 1)

        out = np.full(elevations.shape, np.nan, dtype=np.float32)
        valid = depths >= 0
        out[valid] = arr_values[idx[valid]]

        return out

    def sample(self, elevations: np.ndarray) -> np.ndarray:
        """Sample Vs values at elevations."""
        return self._sample_values(elevations, self.vs)

    def sample_std(self, elevations: np.ndarray) -> np.ndarray:
        """Sample standard deviation values at elevations."""
        return self._sample_values(elevations, self.std)


@dataclass(slots=True, frozen=True)
class VelocitySection:
    velocity_models_1d: tuple[VelocityModel1D, ...]

    def __post_init__(self) -> None:
        if len(self.velocity_models_1d) == 0:
            raise ValueError("velocity_models_1d must contain at least one profile")

        if not all(isinstance(vm, VelocityModel1D) for vm in self.velocity_models_1d):
            raise TypeError(
                "All elements of velocity_models_1d must be instances of VelocityModel1D"
            )

        if not all(vm.label == self.velocity_models_1d[0].label for vm in self.velocity_models_1d):
            raise ValueError("All velocity profiles must have the same label (Vp, Vs, or unknown)")

        xs = [vm.position.x for vm in self.velocity_models_1d]
        if len(xs) != len(set(xs)):
            raise ValueError("All velocity profiles must have unique x coordinates")

        ordered = tuple(
            sorted(
                self.velocity_models_1d,
                key=lambda vm: vm.position.x,
            )
        )
        object.__setattr__(self, "velocity_models_1d", ordered)

    @property
    def xs(self) -> np.ndarray:
        return np.array(
            [vm.position.x for vm in self.velocity_models_1d],
            dtype=np.float32,
        )

    @property
    def topography(self) -> np.ndarray:
        return np.array(
            [vm.position.z for vm in self.velocity_models_1d],
            dtype=np.float32,
        )

    def to_grid(
        self,
        dz: float = 0.01,
        dx: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        profile_xs = np.array(
            [vm.position.x for vm in self.velocity_models_1d],
            dtype=np.float32,
        )

        if dx is None:
            if len(profile_xs) < 2:
                raise ValueError("dx must be given when there is only one profile")
            dx = float(np.min(np.diff(profile_xs)))

        min_thickness = min(float(np.min(vm.thicknesses)) for vm in self.velocity_models_1d)

        if dz > min_thickness:
            raise ValueError(
                f"dz ({dz}) must be smaller than the minimum layer thickness ({min_thickness})"
            )

        tops = [vm.position.z for vm in self.velocity_models_1d]

        bases = [
            vm.position.z
            - float(
                np.sum(
                    np.asarray(
                        vm.thicknesses,
                        dtype=np.float32,
                    )
                )
            )
            for vm in self.velocity_models_1d
        ]

        top = max(tops)
        bottom = min(bases)

        nz = int(np.floor((top - bottom) / dz)) + 1
        zs = (top - np.arange(nz, dtype=np.float32) * dz).astype(np.float32)

        per_profile_vs = np.array(
            [vm.sample(zs) for vm in self.velocity_models_1d],
            dtype=np.float32,
        )

        per_profile_std = np.array(
            [vm.sample_std(zs) for vm in self.velocity_models_1d],
            dtype=np.float32,
        )

        nx = int(np.floor((profile_xs.max() - profile_xs.min()) / dx)) + 1

        xs = (profile_xs.min() + np.arange(nx, dtype=np.float32) * dx).astype(np.float32)

        nearest = np.abs(xs[:, None] - profile_xs[None, :]).argmin(axis=1)

        vs_grid = per_profile_vs[nearest]
        std_grid = per_profile_std[nearest]

        return xs, zs, vs_grid, std_grid
