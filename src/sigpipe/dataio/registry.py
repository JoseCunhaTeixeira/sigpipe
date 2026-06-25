from collections.abc import Callable

from matplotlib.figure import Figure

from sigpipe.base.beamforming import Beam
from sigpipe.base.dispersion_curve import (
    DispersionCurves,
    DispersionCurvesSection,
)
from sigpipe.base.dispersion_image import DispersionImage
from sigpipe.base.inversion import InversionResult
from sigpipe.base.stream import Stream
from sigpipe.base.velocity_model import VelocityModel, VelocityModels, VelocityModelsSection
from sigpipe.dataio.beam.loading import load_beams
from sigpipe.dataio.beam.plotting import plot_beamforming
from sigpipe.dataio.beam.saving import save_beam
from sigpipe.dataio.dispersion.loading import load_dispersion_curves, load_dispersion_image
from sigpipe.dataio.dispersion.plotting import (
    plot_dispersion_curves,
    plot_dispersion_image,
)
from sigpipe.dataio.dispersion.saving import save_dispersion_curves, save_dispersion_image
from sigpipe.dataio.dispersion.section import plot_dispersion_curves_section
from sigpipe.dataio.inversion.plotting import plot_density_curves
from sigpipe.dataio.inversion.saving import save_inversion_result
from sigpipe.dataio.stream.loading import (
    load_gero_active,
    load_gero_passive,
    load_segd,
    load_stream,
)
from sigpipe.dataio.stream.plotting import plot_stream
from sigpipe.dataio.stream.saving import save_stream
from sigpipe.dataio.velocity_model.loading import load_velocity_models
from sigpipe.dataio.velocity_model.plotting import plot_velocity_models
from sigpipe.dataio.velocity_model.saving import save_velocity_model, save_velocity_models
from sigpipe.dataio.velocity_model.section import plot_velocity_models_section


def resolve_handler[R](
    handlers: dict[type, Callable[..., R]], instance: object
) -> Callable[..., R] | None:
    """
    Look up the handler registered for `instance`'s exact type, falling back to
    its base classes (in MRO order) so a handler registered for a base type
    also covers its subclasses unless a more specific override exists.
    """
    for klass in type(instance).__mro__:
        handler = handlers.get(klass)
        if handler is not None:
            return handler
    return None


SAVE_HANDLERS: dict[type, Callable[..., None]] = {
    Stream: save_stream,
    DispersionImage: save_dispersion_image,
    DispersionCurves: save_dispersion_curves,
    VelocityModel: save_velocity_model,
    VelocityModels: save_velocity_models,
    InversionResult: save_inversion_result,
    Beam: save_beam,
}

LOAD_HANDLERS: dict[
    str,
    Callable[..., list[Stream]]
    | Callable[..., list[DispersionImage]]
    | Callable[..., list[DispersionCurves]]
    | Callable[..., list[VelocityModels]]
    | Callable[..., list[Beam]],
] = {
    "stream": load_stream,
    "gero_passive": load_gero_passive,
    "gero_active": load_gero_active,
    "segd": load_segd,
    "dispersion_image": load_dispersion_image,
    "dispersion_curves": load_dispersion_curves,
    "velocity_models": load_velocity_models,
    "beam": load_beams,
}

PLOT_HANDLERS: dict[type, Callable[..., Figure]] = {
    DispersionImage: plot_dispersion_image,
    DispersionCurves: plot_dispersion_curves,
    Stream: plot_stream,
    VelocityModel: plot_velocity_models,
    InversionResult: plot_density_curves,
    Beam: plot_beamforming,
}


SECTION_HANDLERS: dict[type, Callable[..., Figure]] = {
    VelocityModelsSection: plot_velocity_models_section,
    DispersionCurvesSection: plot_dispersion_curves_section,
}
