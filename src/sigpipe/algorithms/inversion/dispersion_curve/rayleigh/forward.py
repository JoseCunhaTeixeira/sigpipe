import numpy as np
from disba import DispersionError, PhaseDispersion

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


def fwd_rayleigh_phase(
    thickness_per_layer: list[float],
    Vs_per_layer: list[float],
    mode: int,
    fs: np.ndarray,
    Vp_Vs_ratio: float,
) -> DispersionCurve:
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
    pd = PhaseDispersion(*velocity_model.T)
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


def fwd_rayleigh_all_modes(
    thickness_per_layer: list[float],
    Vs_per_layer: list[float],
    fs: np.ndarray,
    Vp_Vs_ratio: float,
) -> DispersionCurves | None:
    """Forward-model every Rayleigh mode (0, 1, 2, ...) the model supports, across
    the full given frequency axis, stopping at the first mode disba can't resolve.

    Port of the old Streamlit app's `full_pred_modes` loop in `run_inversion.py`,
    used to overlay all superior modes a model predicts on a dispersion image
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
    pd = PhaseDispersion(*velocity_model.T)

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


def fwd_function(
    state: dict[str, np.ndarray],
    n_layers: int,
    mode: int,
    fs: np.ndarray,
    Vp_Vs_ratio: float,
) -> np.ndarray:
    Vs_per_layer = [state["space"][f"vs{i + 1}"][0] for i in range(n_layers)]
    thickness_per_layer = [state["space"][f"thick{i + 1}"][0] for i in range(n_layers - 1)]
    thickness_per_layer.append(1000)
    dispersion_curve = fwd_rayleigh_phase(thickness_per_layer, Vs_per_layer, mode, fs, Vp_Vs_ratio)
    return dispersion_curve.vs
