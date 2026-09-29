"""The measures of an inverted model and of a soil column: what each describes (the fit, the
chains, the model, the soil column), what it covers, and the limits it is judged against."""

from sigpipe.masw.inversion.measuring import (
    USEFUL_REFERENCE,
    BandFit,
    BoundShare,
    InversionMeasures,
    ModelFit,
)
from sigpipe.masw.quality.model import ModelLimits, measure_model
from sigpipe.masw.quality.soil import SoilLimits, measure_soil


def _fit(model: str, misfits: tuple[float | None, ...], missing: int = 0) -> ModelFit:
    """A fit by band, short wavelengths first (None: no point to weigh in that band)."""
    bands = tuple(
        BandFit(
            wavelength_m=(2.0 * i + 2, 2.0 * i + 4),
            n_points=5,
            misfit=misfit,
            residual=None if misfit is None else 0.01 * misfit,
        )
        for i, misfit in enumerate(misfits)
    )
    known = [misfit for misfit in misfits if misfit is not None]
    return ModelFit(
        model=model, misfit=max(known) if known else None, n_missing=missing, bands=bands
    )


def _measures(**changes: object) -> InversionMeasures:
    """Four chains, watched at two depths: thick1 disagrees, but it is not watched."""
    measures = InversionMeasures(
        fits=(_fit("ensemble", (0.8, None, 1.2)), _fit("median", (0.9, 1.0, 1.1))),
        rhat={"vs@1m": 1.02, "vs@3m": 1.05, "thick1": 1.6},
        watched=("vs@1m", "vs@3m"),
        ess={"vs@1m": 420.0, "vs@3m": 310.0, "thick1": 12.0},
        autocorrelation={"vs@1m": 0.4, "vs@3m": 0.6},
        acceptance=(18.0, 22.0, 26.0, 35.0),
        samples_per_chain=250,
        at_bounds=(BoundShare(parameter="vs2", bound="max", value=500.0, share=0.15),),
        useful_depth_m=6.0,
        useful_reference=USEFUL_REFERENCE,
        depth_max_m=12.0,
        vs_at_depths=((1.0, 150.0), (3.0, 220.0)),
        vs_layers=(150.0, 155.0, 300.0),
        interfaces_m=(2.0, 5.0),
    )
    return measures.model_copy(update=changes)


def test_a_models_measures_say_what_they_describe() -> None:
    report = measure_model(_measures(), ModelLimits(), depth=10.0, free=False)
    measures = {measure.name: measure for measure in report.measures}

    assert {name for name, one in measures.items() if one.of == "fit"} == {
        "misfit_short",
        "misfit_middle",
        "misfit_long",
        "residual_short",
        "residual_middle",
        "residual_long",
        "misfit_layered",
    }
    assert {name for name, one in measures.items() if one.of == "chains"} == {
        "rhat",
        "ess",
        "autocorrelation",
        "acceptance",
        "samples_per_chain",
        "at_bound",
    }
    assert {name for name, one in measures.items() if one.of == "model"} == {
        "depth_informed",
        "contrast",
    }
    assert all(measure.over for measure in report.measures)
    # A band with no point to weigh is not measured: the others judge.
    assert measures["misfit_middle"].value is None and measures["misfit_middle"].passed
    # The chains are judged on the depths watched only.
    assert measures["rhat"].value == 1.05 and measures["ess"].value == 310.0
    assert not measures["at_bound"].passed
    # The layers given: the acceptance reported, the depth informed judged (6 m of 8).
    assert measures["acceptance"].threshold is None
    assert not measures["depth_informed"].passed and not report.informed
    # 150 and 155 m/s are 3.3 % apart: under 5 %, one layer.
    assert measures["contrast"].value == 3.3 and not measures["contrast"].passed


def test_when_the_data_choose_the_layers_the_acceptance_is_judged_in_its_band() -> None:
    report = measure_model(_measures(), ModelLimits(), depth=10.0, free=True)
    acceptance = [one for one in report.measures if one.name == "acceptance"]

    # A floor and a ceiling on the chains' median, 24 %.
    assert [(one.bound, one.threshold, one.passed) for one in acceptance] == [
        ("min", 20.0, True),
        ("max", 30.0, True),
    ]
    # The deepest allowed is the curve's reach: the depth informed reported, not judged.
    (depth,) = [one for one in report.measures if one.name == "depth_informed"]
    assert depth.passed and not report.informed


def test_the_depth_informed_is_left_out_when_read_by_an_older_rule() -> None:
    report = measure_model(_measures(useful_reference="curve"), ModelLimits(), 10.0, False)

    assert "depth_informed" not in {measure.name for measure in report.measures}


def test_a_soil_columns_fit_is_judged_by_band_and_its_water_table_reported() -> None:
    measures = {
        measure.name: measure
        for measure in measure_soil(_fit("soil", (0.5, 2.5, None)), 1.8, SoilLimits())
    }

    assert [name for name, one in measures.items() if one.of == "fit"] == [
        "misfit_short",
        "misfit_middle",
        "misfit_long",
        "residual_short",
        "residual_middle",
        "residual_long",
    ]
    assert not measures["misfit_middle"].passed and measures["misfit_long"].passed
    water = measures["water_table"]
    assert water.of == "soil" and water.value == 1.8 and water.unit == "m"
    assert water.threshold is None and water.passed
    assert all(measure.over for measure in measures.values())
