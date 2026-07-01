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
from sigpipe.transformers.dispersion import Dispersion
from sigpipe.transformers.padding import Pad
from sigpipe.transformers.picking import Pick
from sigpipe.transformers.saving import Save

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
acquisition = PlanarAcquisition(
    source=source, receivers=receivers
)  # Planar acquisition can can (x,y) coordinates but all z must be equal

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


def run() -> None:
    pipeline.run()
