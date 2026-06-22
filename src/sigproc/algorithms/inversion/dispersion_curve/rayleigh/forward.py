import numpy as np
from disba import PhaseDispersion

from sigproc.base.acquisition import UNKNOWN_ACQUISITION
from sigproc.base.dispersion_curve import DispersionCurve, Mode, VelocityType


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
