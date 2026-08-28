from collections.abc import Sequence

import numpy as np
from disba import DispersionError, PhaseDispersion
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
    parsing of `GenerationConfig.under_layers` -- e.g. the string exposed as
    `petro.silex.SilexModel.under_layers` for a loaded Silex checkpoint.
    """
    return tuple(
        UnderLayer(*(float(v) for v in line.split()))
        for line in gpdc_format.splitlines()
        if line.strip()
    )


def _rock_physics_from_petro_model(
    petro_model: PetroModel,
    dz: float,
    kk: int,
    frac: float,
    grain_properties: GrainProperties,
    fluid_properties: FluidProperties,
    g: float,
) -> RockPhysicsResult:
    """Run santiludo's Van Genuchten / Hertz-Mindlin / Biot-Gassmann rock-physics
    chain on a PetroModel, producing a fine (dz-spaced) Vp/Vs/density depth profile."""
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


def _disba_arrays(
    rock_physics: RockPhysicsResult,
    under_layers: Sequence[UnderLayer],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat thickness/Vp/Vs/rho arrays for disba.PhaseDispersion, in (km, km/s, km/s, g/cm3).

    `rock_physics.thks` (`np.diff(np.abs(zs))`) has one fewer element than `VPs`/`VSs`/
    `rhobs` (inter-sample thickness vs. per-sample velocity/density), so it's paired with
    the first `nl` velocity/density samples, dropping the very last one, before appending
    `under_layers` (ordered shallowest to the terminating half-space, thickness 0 on the
    last one). Mirrors (without importing -- it's a private helper) santiludo's own
    `seismic._build_disba_model`.

    Without `under_layers`, the deepest disba "layer" is just the last rock-physics
    sample -- i.e. whatever soil happens to be at the bottom of the modeled column acts
    as the half-space disba extends to infinity. That's rarely physically intended (real
    soil columns sit on a stiffer substrate), so callers comparing against a model that
    assumes a specific substrate -- e.g. Silex, trained with a fixed bedrock under every
    synthetic sample -- must pass the same `under_layers` or the low-frequency end of the
    curve won't correspond.
    """
    nl = len(rock_physics.thks)
    thks = np.concatenate((rock_physics.thks, [layer.thickness for layer in under_layers]))
    vps = np.concatenate((rock_physics.VPs[:nl], [layer.vp for layer in under_layers]))
    vss = np.concatenate((rock_physics.VSs[:nl], [layer.vs for layer in under_layers]))
    rhobs = np.concatenate((rock_physics.rhobs[:nl], [layer.rho for layer in under_layers]))
    return thks / 1_000, vps / 1_000, vss / 1_000, rhobs / 1_000  # m to km and kg/m^3 to g/cm^3


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
    rock_physics = _rock_physics_from_petro_model(
        petro_model, dz, kk, frac, grain_properties, fluid_properties, g
    )
    thks, vps, vss, rhobs = _disba_arrays(rock_physics, under_layers)
    pd = PhaseDispersion(thks, vps, vss, rhobs)
    periods = 1 / fs[::-1]  # Hz to s and reverse
    pd = pd(periods, mode=mode, wave="rayleigh")
    vr = pd.velocity
    if (
        pd.period.shape[0] < periods.shape[0]
    ):  # If the dispersion curve is too short - It is often the case for low velocities (i.e. high periods) on superior modes
        vr = np.append(vr, [np.nan] * (periods.shape[0] - pd.period.shape[0]))
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
    the full given frequency axis, stopping at the first mode disba can't resolve.

    Petro-model counterpart of `rayleigh.forward.fwd_rayleigh_all_modes`, using
    santiludo's rock-physics chain (see `_rock_physics_from_petro_model`) in place of
    a fixed Vp/Vs ratio to get Vp and density. Returns None if not even the
    fundamental mode can be resolved.
    """
    rock_physics = _rock_physics_from_petro_model(
        petro_model, dz, kk, frac, grain_properties, fluid_properties, g
    )
    thks, vps, vss, rhobs = _disba_arrays(rock_physics, under_layers)
    pd = PhaseDispersion(thks, vps, vss, rhobs)

    periods = (1 / fs[fs > 0])[::-1]  # Hz to s and reverse

    curves: list[DispersionCurve] = []
    mode = 0
    while True:
        try:
            data = pd(periods, mode=mode, wave="rayleigh")
        except DispersionError:
            break
        if data.period.shape[0] == 0:
            break
        curves.append(
            DispersionCurve(
                fs=(1 / data.period[::-1]).astype(np.float32),
                vs=(data.velocity[::-1] * 1000).astype(np.float32),
                mode=Mode("R", mode),
                type=VelocityType.PHASE,
                acquisition=UNKNOWN_ACQUISITION,
            )
        )
        mode += 1

    return DispersionCurves(dispersion_curves=tuple(curves)) if curves else None
