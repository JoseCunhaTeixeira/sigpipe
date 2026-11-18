# sigpipe-core

sigpipe's compiled core, in Rust:

- Rayleigh- and Love-wave dispersion curves of layered models: surf96's period equations and
  root search (Computer Programs in Seismology, R. B. Herrmann), the roots refined by Illinois'
  regula falsi;
- the seismic inversion's two samplers: reversible-jump chains with parallel tempering, and
  DREAM(ZS);
- the phase-shift transform's delay and sum.

sigpipe's modules call it (`data.py`, `forward.py`, `transdimensional.py`, `dream.py`,
`phase_shift.py`, and the petrophysical forward model), as does santiludo for its dispersion
curves. `sigpipe_core.phase_dispersion` gives a mode's curve: the periods where the mode exists
and its phase velocities there, raising `sigpipe_core.DispersionError` where the fundamental
mode is not found.

## Installing

Prebuilt wheels, built on Python's stable ABI: one per platform for CPython 3.10 and later, on
Linux (x86_64, aarch64), Windows (x64) and macOS (Apple Silicon, Intel). They are published on
sigpipe's index, which the `pyproject.toml` of sigpipe, santiludo, silex and synthetics name; uv
reads it from there
when PAC or PACo install sigpipe from git, so no Rust toolchain is needed. Elsewhere pip and uv
build the source distribution, which needs Rust (rustup.rs).

## Releasing

`.github/workflows/core-wheels.yml` builds the wheels when a tag `core-v<version>` is pushed,
the version being that of `Cargo.toml` and `pyproject.toml`:

```
git tag core-v0.1.0 && git push origin core-v0.1.0
```

They go to a GitHub Release and to the PEP 503 index on sigpipe's GitHub Pages
(https://josecunhateixeira.github.io/sigpipe/simple/, from the `gh-pages` branch). sigpipe's and
santiludo's `pyproject.toml` then ask for that version, and `uv lock` takes it.

## Developing

With a Rust toolchain, from sigpipe's folder:

```
uv sync
uv pip install ./core          # this checkout's core, over the released one
uv run --no-sync pytest        # --no-sync keeps this checkout's core
cargo test --manifest-path core/Cargo.toml   # the core's own tests
```

## License

MIT ([LICENSE](LICENSE)), except `src/surf96.rs`, under the BSD 3-Clause License of the code it
derives from ([LICENSE-BSD-3-Clause](LICENSE-BSD-3-Clause)).
