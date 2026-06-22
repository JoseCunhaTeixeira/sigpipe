from collections.abc import Callable

from sigproc.base.inversion import InversionResult

from .dispersion_curve.rayleigh.mcmc import inversion_mcmc

DISPERSION_CURVE_INVERSION_METHODS: dict[str, Callable[..., InversionResult]] = {
    "mcmc": inversion_mcmc,
}
