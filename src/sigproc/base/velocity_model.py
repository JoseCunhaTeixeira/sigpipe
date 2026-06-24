from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from sigproc.base.coordinate import Coordinate


@dataclass(slots=True, frozen=True)
class VelocityModel:
    vs_s: tuple[float, ...]
    vs_p: tuple[float, ...]
    rhos: tuple[float, ...]
    vs_s_std: tuple[float, ...]
    thicknesses: tuple[float, ...]
    position: Coordinate

    def __post_init__(self) -> None:
        if not (len(self.thicknesses) == len(self.vs_s) == len(self.vs_s_std)):
            raise ValueError(
                "thicknesses, vs_s, and vs_s_std arrays must have the same length, "
                f"got {len(self.thicknesses)}, {len(self.vs_s)}, and {len(self.vs_s_std)}"
            )

    @property
    def n_layers(self) -> int:
        return len(self.vs_s)

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

    def sample_vs(self, elevations: np.ndarray) -> np.ndarray:
        """Sample Vs values at elevations."""
        return self._sample_values(elevations, self.vs_s)

    def sample_vp(self, elevations: np.ndarray) -> np.ndarray:
        """Sample Vp values at elevations."""
        return self._sample_values(elevations, self.vs_p)

    def sample_rho(self, elevations: np.ndarray) -> np.ndarray:
        """Sample density values at elevations."""
        return self._sample_values(elevations, self.rhos)

    def sample_vs_std(self, elevations: np.ndarray) -> np.ndarray:
        """Sample Vs standard deviation values at elevations."""
        return self._sample_values(elevations, self.vs_s_std)

    def _transition_depths(self) -> np.ndarray:
        """Depths of layer centers and layer-boundary midpoints, from the surface down."""
        depths = [0.0]
        for i, thickness in enumerate(self.thicknesses):
            depths.append(depths[-1] + thickness / 2)
            if i == self.n_layers - 1:
                break
            depths.append(depths[-1] + thickness / 2)
        return np.array(depths, dtype=np.float64)

    def _transition_values(self, values: tuple[float, ...]) -> np.ndarray:
        """Values at layer centers, with boundaries softened to the average of adjacent layers."""
        profile = [values[0]]
        for i in range(self.n_layers):
            profile.append(values[i])
            if i == self.n_layers - 1:
                break
            profile.append((values[i] + values[i + 1]) / 2)
        return np.array(profile, dtype=np.float64)

    def smoothed(self, dz: float, depth_max: float) -> VelocityModel:
        """
        Resample the blocky layered model onto a continuous, uniform-dz depth profile.

        Layer boundaries are softened by averaging adjacent layers, then the
        profile is extended flat down to depth_max and interpolated with a
        shape-preserving PCHIP spline (stays within the local data range --
        unlike a natural cubic spline, it can't ring/overshoot past nearby
        control points on sharp, tightly-spaced layer contrasts).
        """
        depths = self._transition_depths()

        profiles = {
            "vs_s": self._transition_values(self.vs_s),
            "vs_p": self._transition_values(self.vs_p),
            "rhos": self._transition_values(self.rhos),
            "vs_s_std": self._transition_values(self.vs_s_std),
        }

        if depths[-1] < depth_max:
            extra_depths = np.arange(depths[-1] + dz, depth_max, dz)
            extra_depths = np.append(extra_depths, depth_max)
            depths = np.concatenate((depths, extra_depths))
            pad = np.ones_like(extra_depths)
            profiles = {key: np.concatenate((p, pad * p[-1])) for key, p in profiles.items()}

        smooth_depths = np.arange(depths[0], depths[-1], dz)
        smoothed = {key: PchipInterpolator(depths, p)(smooth_depths) for key, p in profiles.items()}

        n = smooth_depths.shape[0]
        return VelocityModel(
            vs_s=tuple(smoothed["vs_s"].tolist()),
            vs_p=tuple(smoothed["vs_p"].tolist()),
            rhos=tuple(smoothed["rhos"].tolist()),
            vs_s_std=tuple(smoothed["vs_s_std"].tolist()),
            thicknesses=tuple([dz] * n),
            position=self.position,
        )


@dataclass(slots=True, frozen=True)
class VelocityModels:
    velocity_models: tuple[VelocityModel, ...]

    def __post_init__(self) -> None:
        if len(self.velocity_models) == 0:
            raise ValueError("At least one velocity model is required")

        if not all(isinstance(vm, VelocityModel) for vm in self.velocity_models):
            raise TypeError("All elements of velocity_models must be instances of VelocityModel")

    def __iter__(self) -> Iterator[VelocityModel]:
        return iter(self.velocity_models)

    def __len__(self) -> int:
        return len(self.velocity_models)

    def __getitem__(self, item: int) -> VelocityModel:
        return self.velocity_models[item]


@dataclass(slots=True, frozen=True)
class VelocityModelsSection(VelocityModels):
    def __post_init__(self) -> None:
        super().__post_init__()

        xs = [vm.position.x for vm in self.velocity_models]
        if len(xs) != len(set(xs)):
            raise ValueError("All velocity profiles must have unique x coordinate")

        ordered = tuple(
            sorted(
                self.velocity_models,
                key=lambda vm: vm.position.x,
            )
        )
        object.__setattr__(self, "velocity_models", ordered)

    @property
    def xs(self) -> np.ndarray:
        return np.array(
            [vm.position.x for vm in self.velocity_models],
            dtype=np.float32,
        )

    @property
    def topography(self) -> np.ndarray:
        return np.array(
            [vm.position.z for vm in self.velocity_models],
            dtype=np.float32,
        )

    def to_grid(
        self,
        dz: float | None,
        dx: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        profile_xs = np.array(
            [vm.position.x for vm in self.velocity_models],
            dtype=np.float32,
        )

        print(profile_xs)

        if dx is None:
            if len(profile_xs) < 2:
                raise ValueError("dx must be given when there is only one profile")
            dx = float(np.min(np.diff(profile_xs))) / 100

        min_thickness = min(float(np.min(vm.thicknesses)) for vm in self.velocity_models)

        if dz is None:
            dz = min_thickness / 100

        if dz > min_thickness:
            raise ValueError(
                f"dz ({dz}) must be smaller than the minimum layer thickness ({min_thickness})"
            )

        tops = [vm.position.z for vm in self.velocity_models]

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
            for vm in self.velocity_models
        ]

        top = max(tops)
        bottom = min(bases)

        nz = int(np.floor((top - bottom) / dz)) + 1
        zs = (top - np.arange(nz, dtype=np.float32) * dz).astype(np.float32)

        per_profile_vs_s = np.array(
            [vm.sample_vs(zs) for vm in self.velocity_models],
            dtype=np.float32,
        )

        per_profile_vs_p = np.array(
            [vm.sample_vp(zs) for vm in self.velocity_models],
            dtype=np.float32,
        )

        per_profile_rhos = np.array(
            [vm.sample_rho(zs) for vm in self.velocity_models],
            dtype=np.float32,
        )

        per_profile_vs_s_std = np.array(
            [vm.sample_vs_std(zs) for vm in self.velocity_models],
            dtype=np.float32,
        )

        nx = int(np.floor((profile_xs.max() - profile_xs.min()) / dx)) + 1

        xs = (profile_xs.min() + np.arange(nx, dtype=np.float32) * dx).astype(np.float32)

        nearest = np.abs(xs[:, None] - profile_xs[None, :]).argmin(axis=1)

        vs_s_grid = per_profile_vs_s[nearest]
        vs_p_grid = per_profile_vs_p[nearest]
        rhos_grid = per_profile_rhos[nearest]
        vs_s_std_grid = per_profile_vs_s_std[nearest]

        return xs, zs, vs_s_grid, vs_p_grid, rhos_grid, vs_s_std_grid
