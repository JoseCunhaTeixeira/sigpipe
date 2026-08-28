from collections.abc import Callable

import numpy as np

from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.base.inversion import InversionResult
from sigpipe.base.petro_model import PetroModel
from sigpipe.base.velocity_model import VelocityModel

from .rayleigh.seismic.forward import fwd_seismic_phase
from .rayleigh.seismic.mcmc import inversion_mcmc


def _inversion_silex(*args: object, **kwargs: object) -> PetroModel:
    # Deferred import: `.petro.silex` requires keras/keras-nlp, an optional
    # dependency (`sigpipe[silex]`). Keeping it out of this module's top level
    # means importing `sigpipe.algorithms`/`sigpipe.transformers` doesn't
    # require keras unless the "silex" method is actually invoked.
    from .rayleigh.petro.silex import inversion_silex

    return inversion_silex(*args, **kwargs)  # type: ignore[arg-type]


DISPERSION_CURVE_INVERSION_METHODS: dict[str, Callable[..., InversionResult | PetroModel]] = {
    "mcmc": inversion_mcmc,
    "silex": _inversion_silex,
}


def _fwd_petro(*args: object, **kwargs: object) -> DispersionCurve:
    # Deferred import: `.petro.forward` requires santiludo, an optional
    # dependency (`sigpipe[santiludo]`). Keeping it out of this module's top
    # level means importing `sigpipe.algorithms`/`sigpipe.transformers`
    # doesn't require santiludo unless a PetroModel is actually forward-modeled.
    from .rayleigh.petro.forward import fwd_petro_phase

    return fwd_petro_phase(*args, **kwargs)  # type: ignore[arg-type]


def _fwd_rayleigh(
    velocity_model: VelocityModel,
    mode: int,
    fs: np.ndarray,
    Vp_Vs_ratio: float = 1.77,
) -> DispersionCurve:
    return fwd_seismic_phase(
        list(velocity_model.thicknesses), list(velocity_model.vs_s), mode, fs, Vp_Vs_ratio
    )


FORWARD_METHODS: dict[type, Callable[..., DispersionCurve]] = {
    PetroModel: _fwd_petro,
    VelocityModel: _fwd_rayleigh,
}
