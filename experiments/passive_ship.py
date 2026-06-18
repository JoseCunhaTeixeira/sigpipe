from pathlib import Path

from sigproc.base import Acquisition, Coordinate
from sigproc.transformers import (
    Apodize,
    Correlate,
    Detrend,
    Filter,
    Load,
    Normalize,
    Plot,
    Slice,
    Stack,
    Whiten,
)

data_dir = Path("//drtfaucon-1/DIN/SMCD/LSPM/Projets/FIND/data/_exemple_donnees_passif")
file_paths = [
    data_dir / "20250311101311_passive_rep0.hdf5",
    data_dir / "20250311101318_passive_rep1.hdf5",
    data_dir / "20250311101324_passive_rep2.hdf5",
    data_dir / "20250311101330_passive_rep3.hdf5",
    data_dir / "20250311101336_passive_rep4.hdf5",
    data_dir / "20250311101342_passive_rep5.hdf5",
    data_dir / "20250311101348_passive_rep6.hdf5",
    data_dir / "20250311101354_passive_rep7.hdf5",
    data_dir / "20250311101400_passive_rep8.hdf5",
    data_dir / "20250311101406_passive_rep9.hdf5",
]

saving_dir = Path("/Users/JC287771/Documents/Work/data/2026-05-29_essai_imagerie_invent/results")

source = Coordinate(x=0, y=0, z=0)
receivers = (
    Coordinate(x=0.110, y=0, z=0),
    Coordinate(x=1.805, y=0, z=0),
    Coordinate(x=0.450, y=0, z=0),
    Coordinate(x=1.855, y=0, z=0),
    Coordinate(x=0.905, y=0, z=0),
    Coordinate(x=2.385, y=0, z=0),
    Coordinate(x=1.405, y=0, z=0),
    Coordinate(x=2.865, y=0, z=0),
)
acquisition = Acquisition(source=source, receivers=receivers)

pipeline = (
    Load(
        file_paths=file_paths,
        data_type="gero_passive",
        sampling_freq=2_000_000,
        acquisition=acquisition,
        key="signals",
        sort=True,
    )
    >> Detrend(method="constant")
    >> Filter(method="iir", fmin=2_000, fmax=8_000, order=4)
    >> Slice(segment_duration=0.003, segment_step=0.003)
    >> Whiten(method="onebit_apod", fmin=2_000, fmax=8_000, taper_width_Hz=1_000)
    >> Normalize(method="onebit")
    >> Apodize(method="hanning", frac=0.1)
    >> Correlate(method="cross", virtual_source_index=0)
    >> Stack(method="phase_weighted", nu=2)
    >> Plot(folder_path=saving_dir, normalize=True)
)


def run() -> None:
    pipeline.run()
