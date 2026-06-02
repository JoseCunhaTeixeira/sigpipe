from src.base.stream import Stream
from src.transformers.appodization import Appodization
from src.transformers.detrending import Detrending
from src.transformers.filtering import Filtering
from src.transformers.loading import Loading
from src.transformers.normalization import Normalization
from src.transformers.slicing import Slicing
from src.transformers.whitening import Whitening

file_paths = []

pipeline = (
    Loading(file_paths=file_paths, data_type=Stream)
    >> Detrending(method="constant")
    >> Detrending(method="linear")
    >> Filtering(method="iir")
    >> Slicing(segment_duration=0.01, segment_step=0.01)
    >> Whitening(method="onebit_apod")
    >> Normalization(method="onebit")
    >> Appodization(method="hanning")
)
