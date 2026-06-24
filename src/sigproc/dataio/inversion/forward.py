from disba import DispersionError

from sigproc.algorithms.inversion.dispersion_curve.rayleigh.forward import fwd_rayleigh_phase
from sigproc.base.dispersion_curve import DispersionCurves
from sigproc.base.inversion import InversionResult

MODEL_NAMES = ("best", "smooth_best", "median", "smooth_median", "ensemble")


def forward_model_all(
    result: InversionResult,
    observed_curves: DispersionCurves,
    Vp_Vs_ratio: float,
) -> dict[str, DispersionCurves | None]:
    """Forward-model every named model (best, smooth_best, median, smooth_median,
    ensemble) at each observed curve's own frequencies.

    Mirrors the old Streamlit app's per-model `.pvc` saves in `run_inversion.py`,
    one dispersion-curve collection per model name. A model is `None` if disba
    can't resolve any mode for it at all (e.g. a finely-discretized smoothed
    model occasionally too ill-conditioned for the fundamental-mode root
    finder); individual unresolved modes are otherwise just omitted.
    """
    models = {name: getattr(result, name) for name in MODEL_NAMES}

    forward_modeled: dict[str, DispersionCurves | None] = {}
    for name, model in models.items():
        curves = []
        for observed_curve in observed_curves:
            try:
                curves.append(
                    fwd_rayleigh_phase(
                        list(model.thicknesses),
                        list(model.vs_s),
                        observed_curve.mode.number,
                        observed_curve.fs,
                        Vp_Vs_ratio,
                    )
                )
            except DispersionError:
                continue
        forward_modeled[name] = DispersionCurves(dispersion_curves=tuple(curves)) if curves else None

    return forward_modeled
