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
    Load(
        file_paths=file_paths,
        data_type="seismic",
        acquisition=acquisition,
        sort=True,
        receivers_to_load=[0, 1, 2, 3, 4, 5, 6],  # Load all traces
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

See [experiments/active_find.py](experiments/active_find.py), [experiments/passive.py](experiments/passive.py), and [experiments/passive_ship.py](experiments/passive_ship.py) for complete worked examples, including dispersion analysis and picking. Run the default experiment with:

```bash
uv run run.py
```
