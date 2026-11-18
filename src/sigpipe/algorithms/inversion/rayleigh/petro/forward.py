from collections.abc import Sequence

import numpy as np
import sigpipe_core
from santiludo import (
    DEFAULT_FLUID_PROPERTIES,
    DEFAULT_GRAIN_PROPERTIES,
    FluidProperties,
    GrainProperties,
    Layer,
    RockPhysicsResult,
    UnderLayer,
    compute_rock_physics,
)

from sigpipe.base.acquisition import UNKNOWN_ACQUISITION
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurves, Mode, VelocityType
from sigpipe.base.petro_model import PetroModel


def parse_under_layers(gpdc_format: str) -> tuple[UnderLayer, ...]:
    """Parse a GPDC-format substratum string (thickness vp vs rho per line,
    one blank-separated quadruplet per line, last line's thickness 0 for the
    terminating half-space) into `UnderLayer`s for `fwd_petro_phase`/
    `fwd_petro_all_modes`. Matches `silex.generation._forward_model`'s own
    parsing of `GenerationConfig.under_layers` -- e.g. the string a Silex
    model's card holds (`petro.silex_catalog.SilexCard.under_layers`).
    """
    return tuple(
        UnderLayer(*(float(v) for v in line.split()))
        for line in gpdc_format.splitlines()
        if line.strip()
    )


def rock_physics(
    petro_model: PetroModel,
    *,
    dz: float = 0.01,
    kk: int = 3,
    frac: float = 0.3,
    grain_properties: GrainProperties = DEFAULT_GRAIN_PROPERTIES,
    fluid_properties: FluidProperties = DEFAULT_FLUID_PROPERTIES,
    g: float = 9.82,
) -> RockPhysicsResult:
    """Run santiludo's Van Genuchten / Hertz-Mindlin / Biot-Gassmann rock-physics
    chain on a PetroModel, producing a fine (dz-spaced) Vp/Vs/density depth profile
    (and the Hertz-Mindlin shear modulus): the chain the forward models below run,
    with the same defaults."""
    layers = [
        Layer(soiltype=str(soil), thickness=thickness, N=float(n), frac=frac)
        for soil, thickness, n in zip(
            petro_model.soils, petro_model.thicknesses, petro_model.Ns, strict=True
        )
    ]
    return compute_rock_physics(
        layers,
        WT=petro_model.water_table_depth,
        dz=dz,
        kk=kk,
        grain_properties=grain_properties,
        fluid_properties=fluid_properties,
        g=g,
    )


def _layer_arrays(
    profile: RockPhysicsResult,
    under_layers: Sequence[UnderLayer],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat thickness/Vp/Vs/rho arrays for sigpipe_core.phase_dispersion, in (km, km/s, km/s,
    g/cm3).

    `profile.thks` (`np.diff(np.abs(zs))`) has one fewer element than `VPs`/`VSs`/
    `rhobs` (inter-sample thickness vs. per-sample velocity/density), so it's paired with
    the first `nl` velocity/density samples, dropping the very last one, before appending
    `under_layers` (ordered shallowest to the terminating half-space, thickness 0 on the
    last one). Mirrors (without importing -- it's a private helper) santiludo's own
    `seismic._layer_arrays`.

    Without `under_layers`, the deepest layer is just the last rock-physics sample -- i.e.
    whatever soil happens to be at the bottom of the modeled column acts as the half-space
    the forward model extends to infinity. That's rarely physically intended (real soil
    columns sit on a stiffer substrate), so callers comparing against a model that assumes a
    specific substrate -- e.g. Silex, trained with a fixed bedrock under every synthetic
    sample -- must pass the same `under_layers` or the low-frequency end of the curve won't
    correspond.
    """
    nl = len(profile.thks)
    thks = np.concatenate((profile.thks, [layer.thickness for layer in under_layers]))
    vps = np.concatenate((profile.VPs[:nl], [layer.vp for layer in under_layers]))
    vss = np.concatenate((profile.VSs[:nl], [layer.vs for layer in under_layers]))
    rhobs = np.concatenate((profile.rhobs[:nl], [layer.rho for layer in under_layers]))

    def per_thousand(values: np.ndarray) -> np.ndarray:
        """m to km and kg/m^3 to g/cm^3, contiguous in float64 as sigpipe_core takes them."""
        return np.ascontiguousarray(values / 1_000, dtype=np.float64)

    return per_thousand(thks), per_thousand(vps), per_thousand(vss), per_thousand(rhobs)


def fwd_petro_phase(
    petro_model: PetroModel,
    mode: int,
    fs: np.ndarray,
    *,
    under_layers: Sequence[UnderLayer] = (),
    dz: float = 0.01,
    kk: int = 3,
    frac: float = 0.3,
    grain_properties: GrainProperties = DEFAULT_GRAIN_PROPERTIES,
    fluid_properties: FluidProperties = DEFAULT_FLUID_PROPERTIES,
    g: float = 9.82,
) -> DispersionCurve:
    profile = rock_physics(
        petro_model,
        dz=dz,
        kk=kk,
        frac=frac,
        grain_properties=grain_properties,
        fluid_properties=fluid_properties,
        g=g,
    )
    layers = _layer_arrays(profile, under_layers)
    periods = np.ascontiguousarray(1 / fs[::-1], dtype=np.float64)  # Hz to s, increasing
    found, vr = sigpipe_core.phase_dispersion(periods, *layers, mode)
    if (
        found.shape[0] < periods.shape[0]
    ):  # If the dispersion curve is too short - It is often the case for low velocities (i.e. high periods) on superior modes
        vr = np.append(vr, [np.nan] * (periods.shape[0] - found.shape[0]))
    vr = vr[::-1] * 1000  # Reverse back and km/s to m/s
    return DispersionCurve(
        fs=fs,
        vs=vr,
        mode=Mode("R", mode),
        type=VelocityType.PHASE,
        acquisition=UNKNOWN_ACQUISITION,
    )


def fwd_petro_all_modes(
    petro_model: PetroModel,
    fs: np.ndarray,
    *,
    under_layers: Sequence[UnderLayer] = (),
    dz: float = 0.01,
    kk: int = 3,
    frac: float = 0.3,
    grain_properties: GrainProperties = DEFAULT_GRAIN_PROPERTIES,
    fluid_properties: FluidProperties = DEFAULT_FLUID_PROPERTIES,
    g: float = 9.82,
) -> DispersionCurves | None:
    """Forward-model every Rayleigh mode (0, 1, 2, ...) the model supports, across
    the full given frequency axis, stopping at the first mode the forward model can't resolve.

    Petro-model counterpart of `seismic.forward.fwd_seismic_all_modes`, using
    santiludo's rock-physics chain (see `rock_physics`) in place of
    a fixed Vp/Vs ratio to get Vp and density. Returns None if not even the
    fundamental mode can be resolved.
    """
    profile = rock_physics(
        petro_model,
        dz=dz,
        kk=kk,
        frac=frac,
        grain_properties=grain_properties,
        fluid_properties=fluid_properties,
        g=g,
    )
    layers = _layer_arrays(profile, under_layers)
    periods = np.ascontiguousarray((1 / fs[fs > 0])[::-1], dtype=np.float64)  # s, increasing

    curves: list[DispersionCurve] = []
    mode = 0
    while True:
        try:
            found, velocity = sigpipe_core.phase_dispersion(periods, *layers, mode)
        except sigpipe_core.DispersionError:
            break
        if found.shape[0] == 0:
            break
        curves.append(
            DispersionCurve(
                fs=(1 / found[::-1]).astype(np.float32),
                vs=(velocity[::-1] * 1000).astype(np.float32),
                mode=Mode("R", mode),
                type=VelocityType.PHASE,
                acquisition=UNKNOWN_ACQUISITION,
            )
        )
        mode += 1

    return DispersionCurves(dispersion_curves=tuple(curves)) if curves else None
