from .appodization import AppodizationTransformer
from .correlation import CorrelationTransformer
from .detrending import DetrendingTransformer
from .filtering import FilteringTransformer
from .normalizing import NormalizationTransformer
from .slicing import SlicingTransformer
from .stacking import StackingTransformer
from .whitening import WhiteningTransformer

TRANSFORMERS = {
    "appodization": AppodizationTransformer,
    "correlation": CorrelationTransformer,
    "detrending": DetrendingTransformer,
    "filtering": FilteringTransformer,
    "normalizing": NormalizationTransformer,
    "slicing": SlicingTransformer,
    "stacking": StackingTransformer,
    "whitening": WhiteningTransformer,
}
