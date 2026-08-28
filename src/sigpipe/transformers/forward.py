from collections.abc import Sequence

import numpy as np

from sigpipe.algorithms.inversion.registry import FORWARD_METHODS
from sigpipe.base.dispersion_curve import DispersionCurve
from sigpipe.base.petro_model import PetroModel
from sigpipe.base.transformer import Transformer
from sigpipe.base.velocity_model import VelocityModel


class Forward(Transformer[PetroModel | VelocityModel, DispersionCurve]):
    """
    Forward-modeling transformer: dispatches each input item to santiludo
    rock physics (PetroModel) or a fixed Vp/Vs ratio (VelocityModel) by its
    own type -- see `FORWARD_METHODS`.
    """

    def __init__(
        self,
        mode: int,
        fs: np.ndarray,
        **params: object,
    ) -> None:
        self.mode = mode
        self.fs = fs
        self.params = params
        """Extra kwargs forwarded to whichever forward function each item's
        type is registered to in FORWARD_METHODS -- under_layers/dz/kk/frac/
        grain_properties/fluid_properties/g for a PetroModel, Vp_Vs_ratio for
        a VelocityModel. A mixed-type sequence must only pass kwargs valid
        for every type present, since params is shared across all items."""

    def transform(self, data: Sequence[PetroModel | VelocityModel]) -> list[DispersionCurve]:
        self.validate_sequence(data, PetroModel, VelocityModel)

        curves: list[DispersionCurve] = []
        for item in data:
            algorithm = FORWARD_METHODS.get(type(item))
            if algorithm is None:
                raise TypeError(
                    f"No forward method registered for {type(item).__name__}. "
                    f"Available: {[t.__name__ for t in FORWARD_METHODS]}"
                )
            curves.append(algorithm(item, self.mode, self.fs, **self.params))

        return curves
