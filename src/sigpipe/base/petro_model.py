from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from sigpipe.base.coordinate import Coordinate


class SoilType(StrEnum):
    NONE = ""
    CLAY = "clay"
    SILT = "silt"
    LOAM = "loam"
    SAND = "sand"


@dataclass(slots=True, frozen=True)
class PetroModel:
    soils: tuple[SoilType, ...]
    thicknesses: tuple[float, ...]
    Ns: tuple[int, ...]
    water_table_depth: float
    position: Coordinate

    def __post_init__(self) -> None:

        if not (len(self.soils) == len(self.thicknesses) == len(self.Ns)):
            raise ValueError(
                "soils, thicknesses, and Ns arrays must have the same length, "
                f"got {len(self.soils)}, {len(self.thicknesses)}, and {len(self.Ns)}"
            )

        if any(t <= 0 for t in self.thicknesses):
            raise ValueError(f"All thicknesses must be > 0, got {self.thicknesses}")

        if self.water_table_depth <= 0:
            raise ValueError(f"water_table_depth must be > 0, got {self.water_table_depth}")

        total_depth = np.sum(self.depths)
        if self.water_table_depth > total_depth:
            raise ValueError(
                "water table depth must be less than total depth, "
                f"got {self.water_table_depth}, and {total_depth}"
            )

    @property
    def n_layers(self) -> int:
        return len(self.soils)

    @property
    def depths(self) -> tuple[float, ...]:
        """Depth of the bottom of each layer (cumulative thickness)."""
        return tuple(float(d) for d in np.cumsum(self.thicknesses))

    def _sample_values(
        self,
        elevations: np.ndarray,
        values: tuple[SoilType, ...] | tuple[int, ...],
        fill_value: SoilType | float,
        dtype: npt.DTypeLike,
    ) -> np.ndarray:
        """
        Piecewise-constant sampling of layer values at given elevations.

        `fill_value` is used above the surface and below this profile's own
        total depth -- a profile is never extended past what it actually
        describes, so a shorter log next to deeper ones in a section reads
        as missing data there, not as a repeat of its last layer.
        """
        elevations = np.asarray(elevations, dtype=np.float32)
        depths = self.position.z - elevations

        bottoms = np.cumsum(np.asarray(self.thicknesses, dtype=np.float32))
        arr_values = np.asarray(values, dtype=dtype)

        idx = np.searchsorted(bottoms, depths, side="right")
        idx = np.minimum(idx, len(arr_values) - 1)

        out = np.full(elevations.shape, fill_value, dtype=dtype)
        valid = (depths >= 0) & (depths <= bottoms[-1])
        out[valid] = arr_values[idx[valid]]

        return out

    def sample_soil(self, elevations: np.ndarray) -> np.ndarray:
        """Sample soil types at elevations."""
        return self._sample_values(elevations, self.soils, SoilType.NONE, object)

    def sample_N(self, elevations: np.ndarray) -> np.ndarray:
        """Sample N values at elevations."""
        return self._sample_values(elevations, self.Ns, np.nan, np.float32)


@dataclass(slots=True, frozen=True)
class PetroModels:
    petro_models: tuple[PetroModel, ...]

    def __post_init__(self) -> None:
        if len(self.petro_models) == 0:
            raise ValueError("At least one petro model is required")

    def __iter__(self) -> Iterator[PetroModel]:
        return iter(self.petro_models)

    def __len__(self) -> int:
        return len(self.petro_models)

    def __getitem__(self, item: int) -> PetroModel:
        return self.petro_models[item]


@dataclass(slots=True, frozen=True)
class PetroModelsSection(PetroModels):
    def __post_init__(self) -> None:
        super().__post_init__()

        xs = [pm.position.x for pm in self.petro_models]
        if len(xs) != len(set(xs)):
            raise ValueError("All petro profiles must have unique x coordinate")

        ordered = tuple(
            sorted(
                self.petro_models,
                key=lambda pm: pm.position.x,
            )
        )
        object.__setattr__(self, "petro_models", ordered)

    @property
    def xs(self) -> np.ndarray:
        return np.array(
            [pm.position.x for pm in self.petro_models],
            dtype=np.float32,
        )

    @property
    def topography(self) -> np.ndarray:
        return np.array(
            [pm.position.z for pm in self.petro_models],
            dtype=np.float32,
        )

    def to_grid(
        self,
        dz: float | None,
        dx: float | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        profile_xs = np.array(
            [pm.position.x for pm in self.petro_models],
            dtype=np.float32,
        )

        if dx is None:
            if len(profile_xs) < 2:
                raise ValueError("dx must be given when there is only one profile")
            dx = float(np.min(np.diff(profile_xs))) / 100

        min_thickness = min(float(np.min(pm.thicknesses)) for pm in self.petro_models)

        if dz is None:
            dz = min_thickness / 100

        if dz > min_thickness:
            raise ValueError(
                f"dz ({dz}) must be smaller than the minimum layer thickness ({min_thickness})"
            )

        tops = [pm.position.z for pm in self.petro_models]

        bases = [
            pm.position.z
            - float(
                np.sum(
                    np.asarray(
                        pm.thicknesses,
                        dtype=np.float32,
                    )
                )
            )
            for pm in self.petro_models
        ]

        top = max(tops)
        bottom = min(bases)

        nz = int(np.floor((top - bottom) / dz)) + 1
        zs = (top - np.arange(nz, dtype=np.float32) * dz).astype(np.float32)

        per_profile_soils = np.array(
            [pm.sample_soil(zs) for pm in self.petro_models],
            dtype=object,
        )

        per_profile_Ns = np.array(
            [pm.sample_N(zs) for pm in self.petro_models],
            dtype=np.float32,
        )

        nx = int(np.floor((profile_xs.max() - profile_xs.min()) / dx)) + 1

        xs = (profile_xs.min() + np.arange(nx, dtype=np.float32) * dx).astype(np.float32)

        nearest = np.abs(xs[:, None] - profile_xs[None, :]).argmin(axis=1)

        soil_grid = per_profile_soils[nearest]
        N_grid = per_profile_Ns[nearest]

        # Water table depths are per-profile scalars, not layered like soils/Ns;
        # convert to elevation so callers can overlay it directly against zs.
        water_table_elevations = np.array(
            [pm.position.z - pm.water_table_depth for pm in self.petro_models],
            dtype=np.float32,
        )
        water_table_profile = water_table_elevations[nearest]

        return xs, zs, soil_grid, N_grid, water_table_profile
