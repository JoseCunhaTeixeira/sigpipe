from collections.abc import Callable

from sigpipe.base.inversion import InversionResult
from sigpipe.base.petro_model import PetroModel

from .dispersion_curve.rayleigh.mcmc import inversion_mcmc


def _inversion_silex(*args: object, **kwargs: object) -> PetroModel:
    # Deferred import: `.petro.silex` requires keras/keras-nlp, an optional
    # dependency (`sigpipe[silex]`). Keeping it out of this module's top level
    # means importing `sigpipe.algorithms`/`sigpipe.transformers` doesn't
    # require keras unless the "silex" method is actually invoked.
    from .dispersion_curve.petro.silex import inversion_silex

    return inversion_silex(*args, **kwargs)  # type: ignore[arg-type]


DISPERSION_CURVE_INVERSION_METHODS: dict[str, Callable[..., InversionResult | PetroModel]] = {
    "mcmc": inversion_mcmc,
    "silex": _inversion_silex,
}
