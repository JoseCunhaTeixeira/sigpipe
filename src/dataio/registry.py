from src.base.dispersion import DispersionCurve, DispersionImage
from src.base.stream import Stream
from src.dataio.dispersion.loading import load_dispersion_curve, load_dispersion_image
from src.dataio.dispersion.plotting import plot_dispersion_curve, plot_dispersion_image
from src.dataio.dispersion.saving import save_dispersion_curve, save_dispersion_image
from src.dataio.stream.loading import load_gero_active, load_gero_passive
from src.dataio.stream.plotting import plot_stream

SAVE_HANDLERS = {
    DispersionImage: save_dispersion_image,
    DispersionCurve: save_dispersion_curve,
}

LOAD_HANDLERS = {
    DispersionImage: load_dispersion_image,
    DispersionCurve: load_dispersion_curve,
    "gero_passive": load_gero_passive,
    "gero_active": load_gero_active,
}


PLOT_HANDLERS = {
    DispersionImage: plot_dispersion_image,
    DispersionCurve: plot_dispersion_curve,
    Stream: plot_stream,
}
