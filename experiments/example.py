from pathlib import Path

from sigpipe.base import Coordinate, PlanarAcquisition
from sigpipe.transformers import (
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

data_dir = Path("/path/to/data/dir/")
file_paths = [
    data_dir / "file1.data",
    data_dir / "file2.data",
    data_dir / "file3.data",
    data_dir / "file4.data",
]

saving_dir = Path("/path/to/saving/dir/")

source = Coordinate(x=0, y=0, z=0)
receivers = (
    Coordinate(x=0, y=0, z=0),
    Coordinate(x=1, y=0, z=0),
    Coordinate(x=2, y=0, z=0),
    Coordinate(x=3, y=0, z=0),
    Coordinate(x=4, y=0, z=0),
    Coordinate(x=5, y=0, z=0),
    Coordinate(x=6, y=0, z=0),
)
acquisition = PlanarAcquisition(source=source, receivers=receivers)

pipeline = (
    Load(
        file_paths=file_paths,
        data_type="gero_passive",
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
)


def run() -> None:
    pipeline.run()
