from pathlib import Path

from src.base.coordinate import Coordinate
from src.base.dispersion import DispersionCurves
from src.base.pipeline import Pipeline
from src.dataio.dispersion.loading import load_dispersion_curves
from src.transformers.correlation import Correlate
from src.transformers.detrending import Detrend
from src.transformers.dispersion import Dispersion
from src.transformers.flipping import Flip
from src.transformers.loading import Load
from src.transformers.mutting import Mute
from src.transformers.padding import Pad
from src.transformers.picking import Pick
from src.transformers.plotting import Plot
from src.transformers.saving import Save
from src.transformers.section import Section
from src.transformers.stacking import Stack

data_dir = Path(
    "//resinosa/PARTAGES_UNITES/DRT_CND/MATER-SHM_2026_BCH/data/20260529_essais_imagerie_invent/data_imagerie"
)
file_paths = [
    data_dir / "20260529_134245_sin_70kHz_3c_hann_50V_sain1.gero",
    data_dir / "20260529_134848_sin_85kHz_3c_hann_50V_sain1.gero",
    data_dir / "20260529_135452_sin_100kHz_3c_hann_50V_sain1.gero",
    data_dir / "20260529_140057_sin_115kHz_3c_hann_50V_sain1.gero",
    data_dir / "20260529_140702_sin_130kHz_3c_hann_50V_sain1.gero",
    data_dir / "20260529_141309_sin_145kHz_3c_hann_50V_sain1.gero",
    data_dir / "20260529_134546_sin_70kHz_5c_hann_50V_sain1.gero",
    data_dir / "20260529_135150_sin_85kHz_5c_hann_50V_sain1.gero",
    data_dir / "20260529_135754_sin_100kHz_5c_hann_50V_sain1.gero",
    data_dir / "20260529_140359_sin_115kHz_5c_hann_50V_sain1.gero",
    data_dir / "20260529_141005_sin_130kHz_5c_hann_50V_sain1.gero",
    data_dir / "20260529_141612_sin_145kHz_5c_hann_50V_sain1.gero",
]

modeled_curves = load_dispersion_curves(
    path=Path(
        "/Users/JC287771/Documents/Work/data/2026-05-29_essai_imagerie_invent/models/disp_1.72mm.txt"
    )
)

SOURCES = (
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
    Coordinate(x=0.108, y=0.000, z=0),  # 09
    Coordinate(x=0.120, y=0.000, z=0),  # 10
    Coordinate(x=0.132, y=0.000, z=0),  # 11
    Coordinate(x=0.144, y=0.000, z=0),  # 12
    Coordinate(x=0.156, y=0.000, z=0),  # 13
    Coordinate(x=0.168, y=0.000, z=0),  # 14
    Coordinate(x=0.180, y=0.000, z=0),  # 15
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
    Coordinate(x=0.108, y=0.012, z=0),  # 25
    Coordinate(x=0.120, y=0.012, z=0),  # 26
    Coordinate(x=0.132, y=0.012, z=0),  # 27
    Coordinate(x=0.144, y=0.012, z=0),  # 28
    Coordinate(x=0.156, y=0.012, z=0),  # 29
    Coordinate(x=0.168, y=0.012, z=0),  # 30
    Coordinate(x=0.180, y=0.012, z=0),  # 31
)
RECEIVERS = (
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
    Coordinate(x=0.108, y=0.000, z=0),  # 09
    Coordinate(x=0.120, y=0.000, z=0),  # 10
    Coordinate(x=0.132, y=0.000, z=0),  # 11
    Coordinate(x=0.144, y=0.000, z=0),  # 12
    Coordinate(x=0.156, y=0.000, z=0),  # 13
    Coordinate(x=0.168, y=0.000, z=0),  # 14
    Coordinate(x=0.180, y=0.000, z=0),  # 15
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
    Coordinate(x=0.108, y=0.012, z=0),  # 25
    Coordinate(x=0.120, y=0.012, z=0),  # 26
    Coordinate(x=0.132, y=0.012, z=0),  # 27
    Coordinate(x=0.144, y=0.012, z=0),  # 28
    Coordinate(x=0.156, y=0.012, z=0),  # 29
    Coordinate(x=0.168, y=0.012, z=0),  # 30
    Coordinate(x=0.180, y=0.012, z=0),  # 31
)


def run_pipeline(acquisitions, sources_to_load, receivers_to_load, folder_path):
    pipeline_load = (
        Load(
            file_paths=file_paths,
            data_type="gero_active",
            acquisitions=acquisitions,
            key="signal",
            sources_to_load=sources_to_load,
            sensors_to_load=receivers_to_load,
        )
        >> Detrend(method="constant")
        >> Mute(vmin=1_000, vmax=2_500, taper=25)
    )

    pipeline_left = Pipeline(
        [
            Correlate(method="cross", virtual_source_index=0, part="causal"),
        ]
    )

    pipeline_right = Correlate(
        method="cross",
        virtual_source_index=-1,
        part="acausal",
    ) >> Flip(axis="space")

    pipeline_compute_disp = (
        Stack(method="phase_weighted", nu=10)
        >> Plot(folder_path=folder_path, normalize=True)
        >> Pad(n=1_000, taper=25)
        >> Dispersion(
            method="phase",
            fmin=0,
            fmax=2_000_000,
            vmin=0,
            vmax=7_000,
        )
        >> Pick(
            fmins=[20_000],
            fmaxs=[200_000],
            vmins=[0],
            vmaxs=[2_500],
            lbdmins=[0.0065],
            lbdmaxs=[0.1],
            names=["A0"],
            return_image=True,
        )
        >> Plot(
            folder_path=folder_path,
            modeled_curves=modeled_curves,
        )
        >> Save(folder_path=folder_path)
    )

    files = pipeline_load.run()
    left_correls = pipeline_left.run(files)
    right_correls = pipeline_right.run(files)
    pipeline_compute_disp.run(left_correls + right_correls)


def run() -> None:

    folder = Path(
        "/Users/JC287771/Documents/Work/data/2026-05-29_essai_imagerie_invent/results"
    )

    # folder_paths = [
    #     folder / "1/",
    #     folder / "2/",
    #     folder / "3/",
    #     folder / "4/",
    #     folder / "5/",
    #     folder / "6/",
    #     folder / "7/",
    #     folder / "8/",
    #     folder / "9/",
    #     folder / "10/",
    #     folder / "11/",
    #     folder / "12/",
    # ]

    # source_windows = [
    #     [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #     [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
    #     [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
    #     [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    # ]

    # receiver_windows = [
    #     [0, 1, 2, 3, 4],
    #     [1, 2, 3, 4, 5],
    #     [2, 3, 4, 5, 6],
    #     [3, 4, 5, 6, 7],
    #     [4, 5, 6, 7, 8],
    #     [5, 6, 7, 8, 9],
    #     [6, 7, 8, 9, 10],
    #     [7, 8, 9, 10, 11],
    #     [8, 9, 10, 11, 12],
    #     [9, 10, 11, 12, 13],
    #     [10, 11, 12, 13, 14],
    #     [11, 12, 13, 14, 15],
    # ]

    # for sources_to_load, receivers_to_load, folder_path in zip(
    #     source_windows, receiver_windows, folder_paths
    # ):
    #     sources = tuple(SOURCES[i] for i in sources_to_load)
    #     receivers = tuple(RECEIVERS[i] for i in receivers_to_load)
    #     acquisitions = [
    #         Acquisition(source=source, receivers=receivers) for source in sources
    #     ]

    #     run_pipeline(
    #         acquisitions=acquisitions,
    #         sources_to_load=sources_to_load,
    #         receivers_to_load=receivers_to_load,
    #         folder_path=folder_path,
    #     )

    pipeline_section = Load(
        file_paths=[folder / "disps.csv"],
        data_type=DispersionCurves,
    ) >> Section(folder_path=folder)
    pipeline_section.run()
