"""Rayleigh-wave dispersion curves of layered Vs models, Vp and density following Vs: the
curves sigpipe draws for the inverted models, computed by sigpipe-core's surf96."""

from collections.abc import Mapping

import numpy as np
import sigpipe_core
from sigpipe_core import DispersionError

from sigpipe.base.acquisition import UNKNOWN_ACQUISITION
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurves, Mode, VelocityType


def vp_rho_from_vs(
    Vs: np.ndarray,
    Vp_Vs_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive Vp and density from Vs using a fixed Vp/Vs ratio (Gardner's relation for rho)."""
    Vp = Vs * Vp_Vs_ratio
    rho = 0.32 * Vp + 0.77 * 1000
    return Vp, rho


def _layer_arrays(
    velocity_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The columns of a layers x (thickness, Vp, Vs, density) array, each contiguous in float64,
    as sigpipe_core takes them."""
    thickness, vp, vs, rho = (
        np.ascontiguousarray(column, dtype=np.float64) for column in velocity_model.T
    )
    return thickness, vp, vs, rho


def fwd_seismic_phase(
    thickness_per_layer: list[float],
    Vs_per_layer: list[float],
    mode: int,
    fs: np.ndarray,
    Vp_Vs_ratio: float,
) -> DispersionCurve:
    """The phase velocities of Rayleigh mode `mode` at frequencies `fs` (m/s, NaN where the mode
    does not exist). Raises DispersionError where the fundamental mode is not found."""
    Vs_per_layer_arr = np.array(Vs_per_layer, dtype=np.float32)
    Vp_per_layer_arr, rho_per_layer_arr = vp_rho_from_vs(Vs_per_layer_arr, Vp_Vs_ratio)
    velocity_model = (
        np.column_stack(
            (
                thickness_per_layer,
                Vp_per_layer_arr,
                Vs_per_layer_arr,
                rho_per_layer_arr,
            )
        )
        / 1_000
    )  # m to km and kg/m^3 to g/cm^3
    periods = np.ascontiguousarray(1 / fs[::-1], dtype=np.float64)  # Hz to s, increasing
    found, vr = sigpipe_core.phase_dispersion(periods, *_layer_arrays(velocity_model), mode)
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


def fwd_seismic_all_modes(
    thickness_per_layer: list[float],
    Vs_per_layer: list[float],
    fs: np.ndarray,
    Vp_Vs_ratio: float,
) -> DispersionCurves | None:
    """Forward-model every Rayleigh mode (0, 1, 2, ...) the model supports, across
    the full given frequency axis, stopping at the first mode the forward model can't resolve.

    Used to overlay all superior modes a model predicts on a dispersion image
    (not just the picked ones). Returns None if not even the fundamental mode
    can be resolved.
    """
    Vs_per_layer_arr = np.array(Vs_per_layer, dtype=np.float32)
    Vp_per_layer_arr, rho_per_layer_arr = vp_rho_from_vs(Vs_per_layer_arr, Vp_Vs_ratio)
    velocity_model = (
        np.column_stack(
            (thickness_per_layer, Vp_per_layer_arr, Vs_per_layer_arr, rho_per_layer_arr)
        )
        / 1_000
    )  # m to km and kg/m^3 to g/cm^3
    layers = _layer_arrays(velocity_model)
    periods = np.ascontiguousarray((1 / fs[fs > 0])[::-1], dtype=np.float64)  # s, increasing

    curves: list[DispersionCurve] = []
    mode = 0
    while True:
        try:
            found, velocity = sigpipe_core.phase_dispersion(periods, *layers, mode)
        except DispersionError:
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


def fwd_function(
    state: dict[str, np.ndarray],
    n_layers: int,
    mode: int,
    fs: np.ndarray,
    Vp_Vs_ratio: float,
    fixed: Mapping[str, float] | None = None,
) -> np.ndarray:
    """The curve of the model in `state`, the values `fixed` (by name: vs1, thick1, ...) taken
    as they are: they are not sampled."""
    fixed = fixed or {}

    def value(name: str) -> float:
        return fixed[name] if name in fixed else float(state["space"][name][0])

    Vs_per_layer = [value(f"vs{i + 1}") for i in range(n_layers)]
    thickness_per_layer = [value(f"thick{i + 1}") for i in range(n_layers - 1)]
    thickness_per_layer.append(1000)
    dispersion_curve = fwd_seismic_phase(thickness_per_layer, Vs_per_layer, mode, fs, Vp_Vs_ratio)
    return dispersion_curve.vs
