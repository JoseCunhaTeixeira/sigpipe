# sigpipe - Signal processing pipeline

[![CI](https://github.com/JoseCunhaTeixeira/sigpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/JoseCunhaTeixeira/sigpipe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](pyproject.toml)

A Python pipeline for processing seismic/acoustic signals, for both active-source and passive acquisitions.

Raw waveforms go in; dispersion curves and inverted velocity models come out. Each processing step is a composable `Transformer`, chained into a `Pipeline` with `>>`.

## Features

- **Loading** — active/passive shot-gathers, generic streams, dispersion images/curves, velocity models, beamforming results.
- **Stream pre-processing** — detrending, padding, muting, flipping, time shifting (a trigger's delay), normalization, spectral whitening, apodization, IIR filtering, slicing.
- **Correlation & stacking** — cross-correlation, bidirectional correlation, active-shot correlation, linear/root/phase-weighted stacking.
- **Beamforming** — cross-beamforming and f-k based receiver selection.
- **Dispersion analysis** — phase-shift and FTAN dispersion imaging; curve picking within bounds (`maximum`), inside a hand-drawn polygon (`lasso`) or automatically, mode after mode (`tracking`).
- **Seismic inversion** — Bayesian inversion of Rayleigh-wave dispersion curves into 1D Vs profiles. Markov chains (MCMC) try many layered models and keep them in proportion to how well they fit the curve; [`disba`](https://github.com/keurfonluu/disba) computes each model's curve. The layers can be:
  - *chosen by the data* (the default): the number of layers is sampled too, so the result uses as many layers as the curve supports (reversible-jump MCMC with parallel tempering);
  - *given*: you set the number of layers and each one's Vs and thickness ranges (DREAM(ZS)).

  See [References](#references).

  The inversion also estimates how noisy the picks really are, and by default a layer's Vs may be at most 20 % lower than the one above (a stiff layer over a much softer one gives false fits).
- **Petrophysical inversion** — AI inversion of Rayleigh-wave dispersion curves to 1D soil models (via [`silex`](https://github.com/josecunhateixeira/silex)).
- **Forward modeling** — 1D velocity or soil models to Rayleigh dispersion curves, via a fixed Vp/Vs ratio or real rock physics (via [`santiludo`](https://github.com/JoseCunhaTeixeira/santiludo)) respectively, dispatched by model type through a single `Forward` transformer.
- **I/O & plotting** — saving/loading and plotting for every data type above, plus section views across multiple acquisitions.
- **MASW** (`sigpipe.masw`) — a line's records to its velocity section: profiles, windows along the line, the processing settings, runs on disk, picks, inversion per window and sections of the line, and the measures of their quality. [PAC](https://github.com/JoseCunhaTeixeira/PAC) (the web application) and [PACo](https://github.com/JoseCunhaTeixeira/PACo) (its AI agent) are built on it.

## Project structure

```
src/sigpipe/
├── base/          # Core domain types: Stream, Acquisition, Coordinate, DispersionCurve(s),
│                  # DispersionImage, VelocityModel(s), PetroModel(s), Beam, Pipeline, Transformer
├── algorithms/    # Pure algorithm implementations, grouped by category, each with a registry
│                  # (apodization, beamforming, correlation, detrending, dispersion, filtering,
│                  # flipping, inversion, mutting, normalization, padding, picking,
│                  # residual_phase, segmentation, selection, shifting, stacking, whitening)
├── transformers/  # Transformer wrappers around the algorithms, used to build pipelines
├── dataio/        # Loading, saving, and plotting for streams, dispersion data,
│                  # velocity models, and beamforming results
└── masw/          # MASW on a line: profiles, windows, presets (the settings schema),
                   # pipelines, runs, picks, inversion (per window, sections), quality measures

experiments/       # Example/exploratory pipelines (active, passive, passive_ship)
run.py             # Entry point running experiments.active_find
```

## Installation

Requires Python 3.14+. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optional extras, each pulled in only where needed (importing `sigpipe.algorithms`/`sigpipe.transformers` doesn't require any of them):

```bash
uv sync --extra santiludo    # petro forward modeling (fwd_petro_phase, Forward on a PetroModel)
uv sync --extra silex        # petrophysical inversion (Invert(method="silex"))
uv sync --extra beamforming  # beamforming transformer
```

`santiludo` is a compiled Cython/C++ package installed from git (not on PyPI) — building it needs a C++ toolchain (on Windows, MSVC via Visual Studio Build Tools' "Desktop development with C++" workload).

## Usage

A pipeline is built by chaining `Transformer` instances with `>>` and running the result:

```python
from sigpipe.transformers import (
    Load,
    Detrend,
    Mute,
    BidirectionalCorrelate,
    Stack,
    Pick,
    Plot,
    Save,
)

pipeline = (
    Load(
        file_paths=file_paths,
        data_type="seismic",
        acquisition=acquisition,
        sort=True,
        receivers_to_load=[0, 1, 2, 3, 4, 5, 6],  # Load all 7 traces
    )
    >> Detrend(method="constant")
    >> Detrend(method="linear")
    >> Filter(method="iir", fmin=10_000, fmax=20_000, order=4)
    >> Slice(segment_duration=0.002, segment_step=0.002)
    >> Whiten(method="onebit_apod", fmin=10_000, fmax=20_000, taper_width_Hz=1_000)
    >> Normalize(method="onebit")
    >> Apodize(method="hanning", frac=0.1)
    >> Correlate(method="cross", virtual_source_index=0)
    >> Stack(method="phase_weighted", nu=2)
    >> Plot(folder_path=saving_dir, normalize=True)
    >> Save(folder_path=saving_dir)
    >> Pad(n=1_000, taper=25)
    >> Dispersion(method="phase", fmin=0, fmax=2_000_000, vmin=0, vmax=7_000)
    >> Pick(
        method="maximum",
        fmins=[20_000],
        fmaxs=[200_000],
        vmins=[0],
        vmaxs=[2_500],
        lbdmins=[0.0065],
        lbdmaxs=[0.1],
        labels=["M0"],
    )
    >> Plot(folder_path=saving_dir)
    >> Save(folder_path=saving_dir)
)

pipeline.run()
```

See [experiments/example.py](experiments/example.py) for a complete worked example.

### One pattern: algorithms in registries, transformers over them

Every processing step follows the same pattern, so that a new method is a function and one line:

1. **An algorithm** is a plain function of one data object and keyword parameters, e.g. `mute(stream, *, tmin, tmax, vmin, vmax, taper)`.
2. **A registry** names the algorithms of a category: `MUTTING_METHODS = {"mute": mute}`.
3. **A transformer** takes a method name and its parameters, and runs the method on every element: `Mute(method="mute", vmin=80, vmax=1500, taper=50)`. `method="none"` passes the data through.
4. **Settings schemas** are generated from the algorithms' signatures (`sigpipe.masw.presets`): a form or an agent validates the same parameters, with the same names, that the functions take.
5. **An algorithm with many or nested parameters** validates them with a pydantic model defined next to it: the tracking picker with `PickingParameters`, the MCMC inversion with `InversionParameters` (its layering and priors, the Vs drop allowed and the chains' effort). The call stays a method and its parameters, `Invert(method="mcmc", free={"max_layers": 6})` or `Invert(method="mcmc", n_layers=3, vs_layers=[...], ...)`, and a form or an agent sends the same JSON.

Data (streams, dispersion images, curves, models) are frozen dataclasses in `sigpipe.base`; parameters, settings and records written to disk are pydantic models.

To add a method, for instance a CNN dispersion picker: write `pick_cnn(image, *, model_path, ...) -> DispersionImage` in `algorithms/picking/dispersion/`, register it in `DISPERSION_PICKING_METHODS`, and it runs as `Pick(method="cnn", model_path=...)` in any pipeline.

## The MASW layer

`sigpipe.masw` takes a line's records to its velocity section, the way PAC and PACo process them. A **profile** is a folder of records with a `receiver_positions.yaml` and, for shots, a `source_positions.yaml`; a **run** writes `<output>/<profile>/<run_id>/`, one `xmid_<x>/` folder per window.

```python
from pathlib import Path

from sigpipe.algorithms.picking.dispersion.tracking import pick_modes
from sigpipe.masw.inversion import InversionParameters, invert_window
from sigpipe.masw.inversion.section import save_section
from sigpipe.masw.picks import save_pick
from sigpipe.masw.runs import find_run, load_image, run_processing
from sigpipe.masw.workspace import Folders

workspace = Folders(input_dir=Path("data/input"), output_dir=Path("data/output"), workers=4)

if __name__ == "__main__":  # the run's worker processes need this guard
    # Process a profile in one of its modes, with a few settings changed from the preset's.
    manifest = run_processing(
        "active_p1", "active", {"masw": {"length": 24, "step": 12}}, workspace
    )
    run_folder = find_run(manifest.run_id, workspace)

    units = [window.folder for window in manifest.windows if window.status == "succeeded"]
    for unit in units:
        image = load_image(run_folder / unit)
        modes = pick_modes(image)  # the automatic picker, with its diagnostics
        if modes and modes[0].curve is not None:
            save_pick(run_folder / unit, image, modes[0].curve)  # M0, in PAC's layout
            invert_window(run_folder / unit, InversionParameters())  # PAC's form defaults

    save_section(run_folder, units)  # the smooth median's Vs(x, z)
```

- `profiles`: `load_profile`, `list_profiles`.
- `windows`: `build_windows`, and the records and traces a run leaves out.
- `presets`: the settings schema: `make_preset`, `resolve_preset`, `method_defaults`.
- `pipelines`: the preprocessing and image pipelines of each mode.
- `runs`: `run_processing`, `find_run`, `list_runs`.
- `picks`: a window's curves in PAC's layout.
- `inversion`: `invert_window`, its `measuring`, its `priors` derived from the curve, and the line's `section`.
- `quality`: the measures of a record's signal, a dispersion image, an M0 pick, and the lateral consistency along the line. Measures only: PACo's quality gates judge them.
- `petro`: the petrophysical inversion of each window (a Silex model predicts the soils, their N values and the water table from the fundamental mode; santiludo's rock physics gives Vs and the shear modulus with depth), the line's sections, and the measures PACo's gates judge. Needs the `silex` and `santiludo` extras. The bundled Silex models, and the band and velocities each was trained on, are in `algorithms/inversion/rayleigh/petro/silex_catalog.py`, which needs neither.

## Development

Install dev dependencies (pytest, ruff, pre-commit) alongside the project:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Enable pre-commit hooks (ruff lint/format + basic hygiene checks) to run automatically on `git commit`:

```bash
uv run pre-commit install
```

CI runs linting, formatting checks, and the test suite on every push and pull request to `main` (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## References

The seismic inversion's samplers (`sigpipe/algorithms/inversion/rayleigh/seismic/`):

- **DREAM(ZS)**, the layers given (`dream.py`):
  - ter Braak, C. J. F., & Vrugt, J. A. (2008). Differential Evolution Markov Chain with snooker updater and fewer chains. *Statistics and Computing*, 18(4), 435–446. https://doi.org/10.1007/s11222-008-9104-9
  - Vrugt, J. A., ter Braak, C. J. F., Diks, C. G. H., Robinson, B. A., Hyman, J. M., & Higdon, D. (2009). Accelerating Markov chain Monte Carlo simulation by differential evolution with self-adaptive randomized subspace sampling. *International Journal of Nonlinear Sciences and Numerical Simulation*, 10(3), 273–290. https://doi.org/10.1515/IJNSNS.2009.10.3.273
  - Vrugt, J. A. (2016). Markov chain Monte Carlo simulation using the DREAM software package: Theory, concepts, and MATLAB implementation. *Environmental Modelling & Software*, 75, 273–316. https://doi.org/10.1016/j.envsoft.2015.08.013
- **Reversible-jump MCMC with parallel tempering**, the layers chosen by the data (`transdimensional.py`):
  - Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. *Biometrika*, 82(4), 711–732. https://doi.org/10.1093/biomet/82.4.711
  - Bodin, T., Sambridge, M., Tkalčić, H., Arroucau, P., Gallagher, K., & Rawlinson, N. (2012). Transdimensional inversion of receiver functions and surface wave dispersion. *Journal of Geophysical Research: Solid Earth*, 117, B02301. https://doi.org/10.1029/2011JB008560 (also the noise level sampled with the model: hierarchical Bayes)
  - Earl, D. J., & Deem, M. W. (2005). Parallel tempering: Theory, applications, and new perspectives. *Physical Chemistry Chemical Physics*, 7(23), 3910–3916. https://doi.org/10.1039/B509983H

## License

[MIT](LICENSE)
