from typing import Callable

from src.base.stream import Stream

from .onebit import whiten_onebit
from .onebit_apod import whiten_onebit_apod
from .savgol import whiten_savgol
from .stft_onebit import whiten_stft_onebit
from .stft_savgol import whiten_stft_savgol

WHITENING_METHODS: dict[str, Callable[..., Stream]] = {
    "onebit": whiten_onebit,
    "onebit_apod": whiten_onebit_apod,
    "savgol": whiten_savgol,
    "stft_onebit": whiten_stft_onebit,
    "stft_savgol": whiten_stft_savgol,
}
