from collections.abc import Sequence
from typing import Literal

from sigproc.algorithms.inversion.registry import DISPERSION_CURVE_INVERSION_METHODS
from sigproc.base.dispersion_curve import DispersionCurves
from sigproc.base.inversion import InversionResult
from sigproc.base.transformer import Transformer


class Invert(Transformer[DispersionCurves, DispersionCurves | InversionResult]):
    """
    Inversion transformer.
    """

    def __init__(
        self,
        method: Literal["none", "mcmc"],
        **params: object,
    ) -> None:
        self.method = method
        self.params = params

    def transform(
        self,
        data: Sequence[DispersionCurves],
    ) -> list[DispersionCurves] | list[InversionResult]:

        self.validate_sequence(data, DispersionCurves)

        if self.method == "none":
            return list(data)

        algorithm = DISPERSION_CURVE_INVERSION_METHODS.get(self.method)
        if algorithm is None:
            raise ValueError(
                f"Unknown inversion method '{self.method}'. "
                f"Available methods: {list(DISPERSION_CURVE_INVERSION_METHODS.keys())}"
            )

        return [
            algorithm(
                dispersion_curves=dispersion_curves,
                position=dispersion_curves[0].acquisition.mid_position,
                **self.params,
            )
            for dispersion_curves in data
        ]
