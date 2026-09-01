from __future__ import annotations

import importlib.machinery
import importlib.resources
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.interpolate import interp1d

os.environ.setdefault("KERAS_BACKEND", "torch")

if "tensorflow" not in sys.modules:
    # keras-hub >=0.31 (what `keras-nlp` now re-exports) unconditionally imports
    # TF-only multimodal model converters from its top-level __init__, even
    # though we only need its 4 core NLP layers below and never touch those
    # converters. Separately, keras's own (backend-agnostic) save/load code
    # lazily probes `tf.TensorShape`/`tf.DType`/`tf.TypeSpec`/`tf.dtypes` to
    # special-case real TensorFlow tensors -- harmless to stub out, since a
    # torch-backend model never produces real instances of those. TensorFlow
    # has no Python 3.14 wheels yet, so without this stub `import keras_nlp`
    # (and keras's own model loading) is simply impossible on this project's
    # Python version. Remove once keras-hub lazy-imports those converters or
    # TensorFlow ships 3.14 wheels.
    class _StubTensorShape:
        pass

    class _StubDType:
        pass

    class _StubTypeSpec:
        pass

    _tf_stub_spec = importlib.machinery.ModuleSpec("tensorflow", loader=None)
    _tf_stub = importlib.util.module_from_spec(_tf_stub_spec)
    setattr(_tf_stub, "TensorShape", _StubTensorShape)  # noqa: B010
    setattr(_tf_stub, "DType", _StubDType)  # noqa: B010
    setattr(_tf_stub, "TypeSpec", _StubTypeSpec)  # noqa: B010
    setattr(_tf_stub, "dtypes", object())  # noqa: B010
    sys.modules["tensorflow"] = _tf_stub

import keras
import keras_nlp  # noqa: F401  # pyright: ignore[reportUnusedImport]  import-only: registers keras-nlp's layers so keras.saving.load_model can resolve them

from sigpipe.base.coordinate import Coordinate
from sigpipe.base.dispersion_curve import DispersionCurve, DispersionCurvesImage, Mode
from sigpipe.base.petro_model import PetroModel, SoilType

_FUNDAMENTAL_RAYLEIGH = Mode("R", 0)

_BUNDLED_MODELS_ROOT = Path(str(importlib.resources.files("sigpipe") / "models" / "silex"))
_REQUIRED_FILES = ("silex.keras", "silex_params.json", "vocab.json")


def list_bundled_silex_models() -> list[str]:
    """Names of every Silex checkpoint bundled with sigpipe itself (see
    pyproject.toml's package-data) -- one per subdirectory of
    `models/silex/` containing `silex.keras`/`silex_params.json`/
    `vocab.json`. Named `<site>_<min_freq>-<max_freq>hz_<min_vel>-<max_vel>mps`
    after the frequency/velocity band each was trained on, since that's what
    determines whether a given model applies to a given survey."""
    if not _BUNDLED_MODELS_ROOT.is_dir():
        return []
    return sorted(
        p.name
        for p in _BUNDLED_MODELS_ROOT.iterdir()
        if p.is_dir() and all((p / f).is_file() for f in _REQUIRED_FILES)
    )


def bundled_silex_model_dir(name: str) -> Path:
    """Resolve a bundled Silex model name (see `list_bundled_silex_models`)
    to its directory."""
    model_dir = _BUNDLED_MODELS_ROOT / name
    if not all((model_dir / f).is_file() for f in _REQUIRED_FILES):
        raise ValueError(
            f"Unknown bundled Silex model {name!r}. Available: {list_bundled_silex_models()}"
        )
    return model_dir


DEFAULT_MODEL_DIR = bundled_silex_model_dir("grand_est_15-50hz_193-415mps")
"""The Silex checkpoint used by `inversion_silex`/`silex_under_layers` when
`model_dir` isn't given, so sigpipe works out of the box even when installed
as a dependency of another project. Pass an explicit `model_dir` (e.g. from
`bundled_silex_model_dir` for a different bundled model, or your own
retrained checkpoint) to use something else instead."""


@dataclass(slots=True, frozen=True)
class SilexModel:
    """Loaded Silex artifact: an encoder-decoder Keras model plus the
    vocab/normalization metadata needed to run it on a dispersion curve."""

    keras_model: keras.Model
    min_freq: float
    max_freq: float
    d_freq: float
    n_freqs: int
    min_vel: float
    max_vel: float
    word_to_index: dict[str, int]
    index_to_word: dict[int, str]
    output_seq_length: int
    forbidden_tokens: tuple[tuple[int, ...], ...]
    """Per-decode-step id blocklist enforcing the trained output grammar
    ([WT] <v> ([SOILi] <soil> [THICKNESSi] <v> [Ni] <v>)* [END]), copied
    verbatim from the export -- see export_legacy_model.py."""
    under_layers: str
    """The fixed substratum every training sample was generated with, in
    GPDC format (thickness vp vs rho per line, last line's thickness 0 for
    the terminating half-space) -- see silex/generation.py's
    GenerationConfig.under_layers. A forward-modeling comparison against a
    curve this model produced/consumed (e.g. a QC round-trip) needs to pass
    the same substratum to `petro.forward.fwd_petro_phase`, or the
    low-frequency end of the curve won't correspond. Kept as this string
    (rather than a list of santiludo.UnderLayer) so loading a Silex
    checkpoint doesn't require santiludo installed -- see
    `petro.forward.parse_under_layers` to convert it."""

    @classmethod
    def load(cls, model_dir: Path) -> SilexModel:
        loaded = keras.saving.load_model(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            model_dir / "silex.keras"
        )
        keras_model = cast("keras.Model", loaded)
        params = json.loads((model_dir / "silex_params.json").read_text())
        vocab = json.loads((model_dir / "vocab.json").read_text())

        return cls(
            keras_model=keras_model,
            min_freq=params["min_freq"],
            max_freq=params["max_freq"],
            d_freq=params["d_freq"],
            n_freqs=params["n_freqs"],
            min_vel=params["min_vel"],
            max_vel=params["max_vel"],
            word_to_index=vocab["word_to_index"],
            index_to_word={int(k): v for k, v in vocab["index_to_word"].items()},
            output_seq_length=params["output_seq_length"],
            forbidden_tokens=tuple(tuple(step) for step in vocab["forbidden_tokens"]),
            under_layers=params["generation_config"]["under_layers"],
        )

    def _preprocess(self, dispersion_curve: DispersionCurve) -> np.ndarray:
        """Resample onto the model's fixed frequency axis and min-max normalize,
        matching the old repo's `misc.resamp` + `run_invertion.py` exactly.

        Raises ValueError if `dispersion_curve` doesn't actually cover the
        model's trained frequency/velocity range: silently extrapolating (a
        curve narrower than the model's frequency band) or normalizing
        outside [0, 1] (velocities outside what the model saw in training)
        produces an unreliable prediction instead of a clear failure.
        """
        fs_min = float(np.nanmin(dispersion_curve.fs))
        fs_max = float(np.nanmax(dispersion_curve.fs))
        if fs_min > self.min_freq or fs_max < self.max_freq:
            raise ValueError(
                f"dispersion_curve frequency range [{fs_min}, {fs_max}] Hz does not cover "
                f"this Silex model's trained range [{self.min_freq}, {self.max_freq}] Hz"
            )

        fs_grid = self.min_freq + np.arange(self.n_freqs) * self.d_freq
        resample = interp1d(
            dispersion_curve.fs,
            dispersion_curve.vs,
            fill_value="extrapolate",  # pyright: ignore[reportArgumentType]
        )
        vs_resampled = np.asarray(resample(fs_grid), dtype=np.float64)

        vs_min = float(np.nanmin(vs_resampled))
        vs_max = float(np.nanmax(vs_resampled))
        if vs_min < self.min_vel or vs_max > self.max_vel:
            raise ValueError(
                f"dispersion_curve velocity range [{vs_min}, {vs_max}] m/s (resampled onto "
                f"this Silex model's frequency axis) falls outside its trained range "
                f"[{self.min_vel}, {self.max_vel}] m/s"
            )

        vs_norm = (vs_resampled - self.min_vel) / (self.max_vel - self.min_vel)
        return vs_norm.reshape(1, self.n_freqs, 1).astype(np.float32)

    def _decode(self, x: np.ndarray) -> list[int]:
        """Greedy, grammar-masked autoregressive decode -- a direct reimplementation
        of `Transformer.decode_seq_restrictive`/`RestrictiveSampler` without keras-nlp's
        Sampler machinery. `forbidden_tokens[i]` masks the logits used to fill prompt
        position `i + 1`, mirroring the old sampler's `index-1`-vs-`index` bookkeeping."""
        end_id = self.word_to_index["[END]"]
        pad_id = self.word_to_index["[PAD]"]

        prompt = np.full((1, self.output_seq_length), pad_id, dtype=np.int32)
        prompt[0, 0] = self.word_to_index["[START]"]

        decoded: list[int] = []
        for i in range(self.output_seq_length - 1):
            # keras's stubs leave Model.__call__/ops.convert_to_numpy's return types
            # too loose for pyright to narrow even with an explicit Any annotation;
            # np.asarray(..., dtype=...) below re-establishes a concrete type for
            # everything downstream.
            logits: Any = self.keras_model([x, prompt], training=False)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            raw_step_logits = keras.ops.convert_to_numpy(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportIndexIssue, reportOptionalSubscript]
                logits[:, i, :]
            )[0]
            step_logits = np.asarray(
                raw_step_logits,  # pyright: ignore[reportUnknownArgumentType]
                dtype=np.float32,
            )
            step_logits[list(self.forbidden_tokens[i])] = -np.inf
            next_id = int(np.argmax(step_logits))
            decoded.append(next_id)
            if next_id == end_id:
                break
            prompt[0, i + 1] = next_id

        return decoded

    def predict(self, dispersion_curve: DispersionCurve, position: Coordinate) -> PetroModel:
        """Predict a PetroModel from one Rayleigh fundamental-mode dispersion curve."""
        x = self._preprocess(dispersion_curve)
        decoded_ids = self._decode(x)
        words = [self.index_to_word[i] for i in decoded_ids]

        # Fixed stride-6-per-layer layout after dropping the leading [START]:
        # words[0:2] = [WT] <value>, then per layer i: [SOILi] <soil>
        # [THICKNESSi] <thickness> [Ni] <N> -- identical strides to
        # run_invertion.py's decoded_GM parsing.
        water_table_depth = float(words[1])
        soil_words = [w for w in words[3::6] if w not in ("[PAD]", "[END]")]
        thickness_words = [w for w in words[5::6] if w not in ("[PAD]", "[END]")]
        n_words = [w for w in words[7::6] if w not in ("[PAD]", "[END]")]

        return PetroModel(
            soils=tuple(SoilType(w) for w in soil_words),
            thicknesses=tuple(float(w) for w in thickness_words),
            Ns=tuple(round(float(w)) for w in n_words),
            water_table_depth=water_table_depth,
            position=position,
        )


@lru_cache(maxsize=4)
def _load_silex_model(model_dir: Path) -> SilexModel:
    return SilexModel.load(model_dir)


def inversion_silex(
    dispersion_curves: DispersionCurvesImage,
    position: Coordinate,
    model_dir: Path | None = None,
) -> PetroModel:
    """Predict a PetroModel from the fundamental-mode Rayleigh curve in
    `dispersion_curves`, using the Silex model artifact at `model_dir`
    (loaded once and cached -- see `_load_silex_model`). Defaults to the
    checkpoint bundled with sigpipe (`DEFAULT_MODEL_DIR`) when not given."""
    curve = next((dc for dc in dispersion_curves if dc.mode == _FUNDAMENTAL_RAYLEIGH), None)
    if curve is None:
        raise ValueError(
            "Silex requires a fundamental-mode Rayleigh dispersion curve (Mode('R', 0))"
        )

    model = _load_silex_model(model_dir if model_dir is not None else DEFAULT_MODEL_DIR)
    return model.predict(curve, position)


def silex_under_layers(model_dir: Path | None = None) -> str:
    """The fixed substratum (GPDC format) the Silex model at `model_dir` was
    trained with -- see `SilexModel.under_layers`. Defaults to the checkpoint
    bundled with sigpipe (`DEFAULT_MODEL_DIR`) when not given. A separate
    accessor rather than folding this into `inversion_silex`'s return value
    keeps that function's `PetroModel` return type matching every other entry
    in `DISPERSION_CURVE_INVERSION_METHODS`; `_load_silex_model`'s cache means
    calling this alongside `inversion_silex` for the same `model_dir` doesn't
    reload anything."""
    return _load_silex_model(model_dir if model_dir is not None else DEFAULT_MODEL_DIR).under_layers
