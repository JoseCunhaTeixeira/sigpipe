"""The measures of a soil column (PACo's petrophysical QC, G7, judges them; PAC measures its own
runs' alike): the curve the column gives back against the pick, by band of wavelength as a
seismic model's (the fit), and its water table (the soil column). Each measure says what it
describes and what it covers."""

from pydantic import BaseModel, ConfigDict, Field

from sigpipe.masw.inversion.measuring import ModelFit
from sigpipe.masw.quality.measures import Measure, figure
from sigpipe.masw.quality.model import BAND_NAMES


class SoilLimits(BaseModel):
    """How a soil column is measured, and its limits (G5's, provisional)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    max_misfit: float = Field(
        default=2.0,
        gt=0,
        description="RMS of the residuals over the curve's uncertainties, in any band, at most: "
        "about 1 is a fit within the errors.",
    )
    n_bands: int = Field(default=3, ge=1, description="Bands of wavelength the fit is judged in.")


def measure_soil(
    fit: ModelFit, water_table_m: float | None, limits: SoilLimits
) -> tuple[Measure, ...]:
    """A soil column's measures against `limits`, each saying what it covers: its curve's fit
    by band (a band with no point to weigh is not measured: the others judge), their residuals,
    its water table (reported). One definition for PACo's G7 and PAC's views."""
    names = BAND_NAMES.get(len(fit.bands), tuple(f"band{i + 1}" for i in range(len(fit.bands))))

    def over(n_points: int, wavelengths: tuple[float, float]) -> str:
        return (
            f"the {n_points} picked points at {figure(wavelengths[0])}-"
            f"{figure(wavelengths[1])} m of wavelength, against the soil column's curve"
        )

    measures = [
        Measure(
            name=f"misfit_{name}",
            value=band.misfit,
            threshold=limits.max_misfit,
            bound="max",
            passed=band.misfit is None or band.misfit <= limits.max_misfit,
            of="fit",
            over=over(band.n_points, band.wavelength_m),
        )
        for name, band in zip(names, fit.bands, strict=True)
    ]
    measures += [
        Measure(
            name=f"residual_{name}",
            value=None if band.residual is None else round(100 * band.residual, 1),
            passed=True,
            unit="%",
            of="fit",
            over=over(band.n_points, band.wavelength_m)
            + ": (modelled - picked) / modelled, median",
        )
        for name, band in zip(names, fit.bands, strict=True)
    ]
    measures.append(
        Measure(
            name="water_table",
            value=water_table_m,
            passed=True,
            unit="m",
            of="soil",
            over="the soil column's depth to water",
        )
    )
    return tuple(measures)
