# sigpipe

A Python pipeline for processing seismic/acoustic signals and estimating subsurface velocity models from surface-wave dispersion, for both active-source (MASW-style) and passive (ambient-noise) acquisitions.

Raw waveforms go in; dispersion curves and inverted velocity models come out. Each processing step is a composable `Transformer`, chained into a `Pipeline` with `>>`.

## Features

- **Loading** — active/passive shot-gathers, generic streams, dispersion images/curves, velocity models, beamforming results.
- **Stream pre-processing** — detrending, padding, muting, flipping, normalization, spectral whitening, apodization, IIR filtering, slicing.
- **Correlation & stacking** — cross-correlation, bidirectional correlation, active-shot correlation, linear/root/phase-weighted stacking.
- **Beamforming** — cross-beamforming and f-k based receiver selection.
- **Dispersion analysis** — phase-shift and FTAN dispersion imaging, automated/manual curve picking.
- **Inversion** — Bayesian MCMC inversion of Rayleigh-wave dispersion curves to 1D velocity models (via `disba` + `bayesbay`).
- **I/O & plotting** — saving/loading and plotting for every data type above, plus section views across multiple acquisitions.

## Project structure

```
src/sigpipe/
├── base/          # Core domain types: Stream, Acquisition, Coordinate, DispersionCurve(s),
│                  # DispersionImage, VelocityModel(s), Beam, Pipeline, Transformer
├── algorithms/    # Pure algorithm implementations, grouped by category, each with a registry
│                  # (apodization, beamforming, correlation, detrending, dispersion, filtering,
│                  # flipping, inversion, mutting, normalization, padding, picking,
│                  # segmentation, selection, stacking, whitening)
├── transformers/  # Transformer wrappers around the algorithms, used to build pipelines
└── dataio/        # Loading, saving, and plotting for streams, dispersion data,
                   # velocity models, and beamforming results

experiments/       # Example/exploratory pipelines (active, passive, passive_ship)
run.py             # Entry point running experiments.active_find
```

## Installation

Requires Python 3.14. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

A pipeline is built by chaining `Transformer` instances with `>>` and running the result:

```python
from sigpipe.transformers import Load, Detrend, Mute, BidirectionalCorrelate, Stack, Pick, Plot, Save

pipeline = (
    Load(file_paths=file_paths, data_type="seismic", acquisitions=acquisitions)
    >> Detrend(method="constant")
    >> Mute(vmin=1_000, vmax=2_500, taper=25)
    >> BidirectionalCorrelate(method="cross")
    >> Stack(method="phase_weighted", nu=2)
    >> Pick(method="maximum")
    >> Plot(folder_path=folder_path, normalize=True)
    >> Save(folder_path=folder_path)
)

pipeline.run()
```

See [experiments/active_find.py](experiments/active_find.py), [experiments/passive.py](experiments/passive.py), and [experiments/passive_ship.py](experiments/passive_ship.py) for complete worked examples, including dispersion analysis and picking. Run the default experiment with:

```bash
uv run run.py
```
