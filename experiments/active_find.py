from pathlib import Path

from src.base.acquisition import Acquisition
from src.base.coordinate import Coordinate
from src.dataio.dispersion.loading import load_dispersion_curve
from src.transformers.detrending import Detrend
from src.transformers.dispersion import Dispersion
from src.transformers.loading import Load
from src.transformers.padding import Pad
from src.transformers.plotting import Plot

data_dir = Path(
    "//resinosa/PARTAGES_UNITES/DRT_CND/MATER-SHM_2026_BCH/data/20260529_essais_imagerie_invent/data_imagerie"
)
file_paths = [
    data_dir / "20260529_135754_sin_100kHz_5c_hann_50V_sain1.gero",
]

saving_folder_path = Path(
    "/Users/JC287771/Documents/Work/data/2026-05-29_essai_imagerie_invent/results"
)

modeled_curves = load_dispersion_curve(
    path=Path(
        "/Users/JC287771/Documents/Work/data/2026-05-29_essai_imagerie_invent/models/disp_1.72mm.txt"
    )
)

sources = (
    # 1st ring
    Coordinate(x=0.000, y=0.000, z=0),  # 00
    Coordinate(x=0.012, y=0.000, z=0),  # 01
    Coordinate(x=0.024, y=0.000, z=0),  # 02
    Coordinate(x=0.036, y=0.000, z=0),  # 03
    Coordinate(x=0.048, y=0.000, z=0),  # 04
    Coordinate(x=0.060, y=0.000, z=0),  # 05
    Coordinate(x=0.072, y=0.000, z=0),  # 06
    Coordinate(x=0.084, y=0.000, z=0),  # 07
    Coordinate(x=0.096, y=0.000, z=0),  # 08
    Coordinate(x=-0.084, y=0.000, z=0),  # 09
    Coordinate(x=-0.072, y=0.000, z=0),  # 10
    Coordinate(x=-0.060, y=0.000, z=0),  # 11
    Coordinate(x=-0.048, y=0.000, z=0),  # 12
    Coordinate(x=-0.036, y=0.000, z=0),  # 13
    Coordinate(x=-0.024, y=0.000, z=0),  # 14
    Coordinate(x=-0.012, y=0.000, z=0),  # 15
    # 2nd ring
    Coordinate(x=0.000, y=0.012, z=0),  # 16
    Coordinate(x=0.012, y=0.012, z=0),  # 17
    Coordinate(x=0.024, y=0.012, z=0),  # 18
    Coordinate(x=0.036, y=0.012, z=0),  # 19
    Coordinate(x=0.048, y=0.012, z=0),  # 20
    Coordinate(x=0.060, y=0.012, z=0),  # 21
    Coordinate(x=0.072, y=0.012, z=0),  # 22
    Coordinate(x=0.084, y=0.012, z=0),  # 23
    Coordinate(x=0.096, y=0.012, z=0),  # 24
    Coordinate(x=-0.084, y=0.012, z=0),  # 25
    Coordinate(x=-0.072, y=0.012, z=0),  # 26
    Coordinate(x=-0.060, y=0.012, z=0),  # 27
    Coordinate(x=-0.048, y=0.012, z=0),  # 28
    Coordinate(x=-0.036, y=0.012, z=0),  # 29
    Coordinate(x=-0.024, y=0.012, z=0),  # 30
    Coordinate(x=-0.012, y=0.012, z=0),  # 31
)
receivers = (
    # 1st ring
    Coordinate(x=0.000, y=0.000, z=0),  # 00
    Coordinate(x=0.012, y=0.000, z=0),  # 01
    Coordinate(x=0.024, y=0.000, z=0),  # 02
    Coordinate(x=0.036, y=0.000, z=0),  # 03
    Coordinate(x=0.048, y=0.000, z=0),  # 04
    Coordinate(x=0.060, y=0.000, z=0),  # 05
    Coordinate(x=0.072, y=0.000, z=0),  # 06
    Coordinate(x=0.084, y=0.000, z=0),  # 07
    Coordinate(x=0.096, y=0.000, z=0),  # 08
    Coordinate(x=-0.084, y=0.000, z=0),  # 09
    Coordinate(x=-0.072, y=0.000, z=0),  # 10
    Coordinate(x=-0.060, y=0.000, z=0),  # 11
    Coordinate(x=-0.048, y=0.000, z=0),  # 12
    Coordinate(x=-0.036, y=0.000, z=0),  # 13
    Coordinate(x=-0.024, y=0.000, z=0),  # 14
    Coordinate(x=-0.012, y=0.000, z=0),  # 15
    # 2nd ring
    Coordinate(x=0.000, y=0.012, z=0),  # 16
    Coordinate(x=0.012, y=0.012, z=0),  # 17
    Coordinate(x=0.024, y=0.012, z=0),  # 18
    Coordinate(x=0.036, y=0.012, z=0),  # 19
    Coordinate(x=0.048, y=0.012, z=0),  # 20
    Coordinate(x=0.060, y=0.012, z=0),  # 21
    Coordinate(x=0.072, y=0.012, z=0),  # 22
    Coordinate(x=0.084, y=0.012, z=0),  # 23
    Coordinate(x=0.096, y=0.012, z=0),  # 24
    Coordinate(x=-0.084, y=0.012, z=0),  # 25
    Coordinate(x=-0.072, y=0.012, z=0),  # 26
    Coordinate(x=-0.060, y=0.012, z=0),  # 27
    Coordinate(x=-0.048, y=0.012, z=0),  # 28
    Coordinate(x=-0.036, y=0.012, z=0),  # 29
    Coordinate(x=-0.024, y=0.012, z=0),  # 30
    Coordinate(x=-0.012, y=0.012, z=0),  # 31
)
sources_to_load = [
    0,
    # 1,
    # 2,
    # 3,
    # 4,
    # 5,
    # 6,
    # 7,
    # 8,
    # 9,
    # 10,
    # 11,
    # 12,
    # 13,
    # 14,
    # 15,
    # 16,
    # 17,
    # 18,
    # 19,
    # 20,
    # 21,
    # 22,
    # 23,
    # 24,
    # 25,
    # 26,
    # 27,
    # 28,
    # 29,
    # 30,
    # 31,
]
receivers_to_load = [
    # 0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    # 8,
    # 9,
    # 10,
    # 11,
    # 12,
    # 13,
    # 14,
    # 15,
    # 16,
    # 17,
    # 18,
    # 19,
    # 20,
    # 21,
    # 22,
    # 23,
    # 24,
    # 25,
    # 26,
    # 27,
    # 28,
    # 29,
    # 30,
    # 31,
]

sources = tuple(sources[i] for i in sources_to_load)
receivers = tuple(receivers[i] for i in receivers_to_load)
acquisitions = [Acquisition(source=source, receivers=receivers) for source in sources]

pipeline = (
    Load(
        file_paths=file_paths,
        data_type="gero_active",
        acquisitions=acquisitions,
        key="signal",
        sources_to_load=sources_to_load,
        sensors_to_load=receivers_to_load,
    )
    >> Detrend(method="constant")
    >> Plot(folder_path=saving_folder_path, normalize=True)
    >> Pad(n=1_000, taper=100)
    >> Dispersion(
        method="phase",
        fmin=10_000,
        fmax=1_000_000,
        vmin=0,
        vmax=7_000,
        vmin_expected=200,
    )
    >> Plot(
        folder_path=saving_folder_path,
        normalize=True,
        modeled_curves=modeled_curves,
    )
)
