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

data_dir = Path("//drtvini/Echange/JCU/exemple donnees passives/data_passif_sain")
file_paths = [
    data_dir / "20250328_162747_1_1.gero",
    data_dir / "20250328_162751_1_2.gero",
    data_dir / "20250328_162756_1_3.gero",
    data_dir / "20250328_162800_1_4.gero",
    data_dir / "20250328_162804_1_5.gero",
]

saving_dir = Path(
    "/Users/JC287771/Documents/Work/data/2026-05-09_passive_tests/results"
)

source = Coordinate(x=0, y=0, z=0)
receivers = (
    ### PZT
    Coordinate(x=0, y=0, z=0),
    Coordinate(x=1, y=0, z=0),
    Coordinate(x=2, y=0, z=0),
    Coordinate(x=3, y=0, z=0),
    Coordinate(x=-3, y=0, z=0),
    Coordinate(x=-2, y=0, z=0),
    Coordinate(x=-1, y=0, z=0),
    ### FIBER
    Coordinate(x=0, y=0, z=0),
    Coordinate(x=1, y=0, z=0),
    Coordinate(x=2, y=0, z=0),
    Coordinate(x=3, y=0, z=0),
    Coordinate(x=-3, y=0, z=0),
    Coordinate(x=-2, y=0, z=0),
    Coordinate(x=-1, y=0, z=0),
)
receivers_to_load = [
    ### PZT
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    ### FIBER
    # 7,
    # 8,
    # 9,
    # 10,
    # 11,
    # 12,
    # 13,
]
receivers = tuple(receivers[i] for i in receivers_to_load)
acquisition = Acquisition(source=source, receivers=receivers)

pipeline = (
    Load(
        file_paths=file_paths,
        data_type="gero_passive",
        acquisition=acquisition,
        sort=True,
        receivers_to_load=receivers_to_load,
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
