from pathlib import Path

from src.transformers.appodization import Appodization
from src.transformers.correlation import Correlation
from src.transformers.detrending import Detrending
from src.transformers.filtering import Filtering
from src.transformers.loading import Loading
from src.transformers.normalization import Normalization
from src.transformers.plotting import Plotting
from src.transformers.slicing import Slicing
from src.transformers.stacking import Stacking
from src.transformers.whitening import Whitening

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

saving_folder_path = Path(
    "/Users/JC287771/Documents/Work/data/2026-05-29_essai_imagerie_invent/results"
)

pipeline = (
    Loading(
        file_paths=file_paths,
        data_type="gero_passive",
        key="signals",
        sampling_freq=2_000_000,
    )
    >> Detrending(method="constant")
    >> Filtering(method="iir", fmin=2_000, fmax=8_000, order=4)
    >> Slicing(segment_duration=0.003, segment_step=0.003)
    >> Whitening(method="onebit_apod", fmin=2_000, fmax=8_000, taper_width_Hz=1_000)
    >> Normalization(method="onebit")
    >> Appodization(method="hanning", frac=0.1)
    >> Correlation(method="cross", virtual_source_index=0)
    >> Stacking(method="phase_weighted", nu=2)
    >> Plotting(folder_path=saving_folder_path, normalize="per_trace")
)
