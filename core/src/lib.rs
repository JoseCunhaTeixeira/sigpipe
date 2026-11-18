//! sigpipe-core: sigpipe's compiled core. Rayleigh- and Love-wave dispersion curves of layered
//! models, the seismic inversion's two samplers, and the phase-shift transform's sum, for
//! sigpipe's Python modules (data.py, forward.py, transdimensional.py, dream.py, phase_shift.py)
//! and for santiludo. Every function releases the GIL while it computes.

mod data;
mod dream;
mod phase_shift;
mod rng;
mod surf96;
mod transdimensional;

use numpy::ndarray::Array2;
use numpy::{
    Complex32, Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

create_exception!(
    sigpipe_core,
    DispersionError,
    PyException,
    "The fundamental mode of a layered model was not found over the periods asked."
);

fn wave_of(wave: &str) -> PyResult<surf96::Wave> {
    match wave {
        "rayleigh" => Ok(surf96::Wave::Rayleigh),
        "love" => Ok(surf96::Wave::Love),
        _ => Err(PyValueError::new_err(format!(
            "wave must be 'rayleigh' or 'love', not {wave:?}"
        ))),
    }
}

/// A dispersion curve as phase_dispersion returns it: the periods (s) and phase velocities (km/s).
type Dispersion<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// A curve as the Python side sends it: its mode (0: the fundamental), then its periods (s,
/// increasing), picked velocities and their uncertainties (m/s).
type CurveArgs<'py> = (
    usize,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
);

fn curves_from(curves: &[CurveArgs]) -> PyResult<Vec<data::Curve>> {
    curves
        .iter()
        .map(|(mode, periods, observed, sigma)| {
            let curve = data::Curve {
                mode: *mode,
                periods: periods.as_slice()?.to_vec(),
                observed: observed.as_slice()?.to_vec(),
                sigma: sigma.as_slice()?.to_vec(),
            };
            if curve.periods.is_empty()
                || curve.periods.len() != curve.observed.len()
                || curve.sigma.len() != curve.observed.len()
            {
                return Err(PyValueError::new_err(
                    "a curve's periods, velocities and uncertainties must be as many, at least one",
                ));
            }
            Ok(curve)
        })
        .collect()
}

fn matrix<'py>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Array2::from_shape_vec((rows, columns), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn layers_of<'a>(
    thickness: &'a PyReadonlyArray1<f64>,
    vp: &'a PyReadonlyArray1<f64>,
    vs: &'a PyReadonlyArray1<f64>,
    rho: &'a PyReadonlyArray1<f64>,
) -> PyResult<[&'a [f64]; 4]> {
    let layers = [
        thickness.as_slice()?,
        vp.as_slice()?,
        vs.as_slice()?,
        rho.as_slice()?,
    ];
    let count = layers[0].len();
    if count == 0 || layers.iter().any(|values| values.len() != count) {
        return Err(PyValueError::new_err(
            "the layers' thicknesses, velocities and densities must be as many, at least one",
        ));
    }
    Ok(layers)
}

/// The phase velocities (km/s) of Rayleigh mode `mode` at `periods` (s, increasing) of the
/// layered model (km, km/s, g/cm3, top down; the last layer a half-space), 0 where a higher
/// mode does not exist; None where the fundamental mode is not found. The roots are bracketed
/// by steps of `dc` (km/s).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rayleigh_phase<'py>(
    py: Python<'py>,
    periods: PyReadonlyArray1<'py, f64>,
    thickness: PyReadonlyArray1<'py, f64>,
    vp: PyReadonlyArray1<'py, f64>,
    vs: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    mode: usize,
    dc: f64,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let t = periods.as_slice()?;
    let [d, a, b, r] = layers_of(&thickness, &vp, &vs, &rho)?;
    let found = py.detach(|| surf96::rayleigh_phase(t, d, a, b, r, mode, dc));
    Ok(found.ok().map(|c| c.into_pyarray(py)))
}

/// The dispersion curve of `wave`'s mode `mode` (0: the fundamental) over `periods` (s,
/// increasing) of the layered model (km, km/s, g/cm3, top down; the last layer a half-space):
/// the periods where the mode exists and its phase velocities there (km/s). The roots are
/// bracketed by steps of `dc` (km/s). Raises DispersionError where the fundamental mode is not
/// found.
#[pyfunction]
#[pyo3(signature = (periods, thickness, vp, vs, rho, mode=0, wave="rayleigh", dc=0.005))]
#[allow(clippy::too_many_arguments)]
fn phase_dispersion<'py>(
    py: Python<'py>,
    periods: PyReadonlyArray1<'py, f64>,
    thickness: PyReadonlyArray1<'py, f64>,
    vp: PyReadonlyArray1<'py, f64>,
    vs: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    mode: usize,
    wave: &str,
    dc: f64,
) -> PyResult<Dispersion<'py>> {
    let t = periods.as_slice()?;
    if t.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(PyValueError::new_err(
            "the periods must be sorted, increasing",
        ));
    }
    let [d, a, b, r] = layers_of(&thickness, &vp, &vs, &rho)?;
    let wave = wave_of(wave)?;
    let c = py
        .detach(|| surf96::phase_velocities(t, d, a, b, r, mode, wave, dc))
        .map_err(|_| DispersionError::new_err("the fundamental mode was not found"))?;
    let (found, velocities): (Vec<f64>, Vec<f64>) = t
        .iter()
        .zip(&c)
        .filter(|&(_, &v)| v > 0.0)
        .map(|(&p, &v)| (p, v))
        .unzip();
    Ok((found.into_pyarray(py), velocities.into_pyarray(py)))
}

/// `wave`'s period equation at phase velocity `c` (km/s) and `period` (s) of the layered model:
/// a check of the roots found.
#[pyfunction]
#[pyo3(signature = (period, c, thickness, vp, vs, rho, wave="rayleigh"))]
fn period_equation(
    period: f64,
    c: f64,
    thickness: PyReadonlyArray1<f64>,
    vp: PyReadonlyArray1<f64>,
    vs: PyReadonlyArray1<f64>,
    rho: PyReadonlyArray1<f64>,
    wave: &str,
) -> PyResult<f64> {
    let [d, a, b, r] = layers_of(&thickness, &vp, &vs, &rho)?;
    Ok(surf96::period_equation(
        wave_of(wave)?,
        period,
        c,
        d,
        a,
        b,
        r,
    ))
}

/// One reversible-jump chain and its hotter copies (transdimensional.py's run_chain), from the
/// curves, the priors and the settings: its kept models, counts and steps, in a dict.
#[pyfunction]
#[pyo3(signature = (
    curves, vs_bounds, depth_bounds, max_layers, least_ratio, vp_vs, noise_bounds, thinnest,
    n_iterations, n_burnin, save_every, temperatures, hottest, seed,
))]
#[allow(clippy::too_many_arguments)]
fn transdimensional_chain<'py>(
    py: Python<'py>,
    curves: Vec<CurveArgs<'py>>,
    vs_bounds: (f64, f64),
    depth_bounds: (f64, f64),
    max_layers: usize,
    least_ratio: f64,
    vp_vs: f64,
    noise_bounds: (f64, f64),
    thinnest: f64,
    n_iterations: usize,
    n_burnin: usize,
    save_every: usize,
    temperatures: usize,
    hottest: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if max_layers < 1 || save_every < 1 {
        return Err(PyValueError::new_err(
            "max_layers and save_every must be at least 1",
        ));
    }
    let space = transdimensional::Space {
        curves: curves_from(&curves)?,
        vs_bounds,
        depth_bounds,
        max_layers,
        least_ratio,
        vp_vs,
        noise_bounds,
        thinnest,
    };
    let settings = transdimensional::Settings {
        n_iterations,
        n_burnin,
        save_every,
        temperatures,
        hottest,
    };
    let chain = py
        .detach(|| transdimensional::run_chain(&space, &settings, seed))
        .map_err(PyValueError::new_err)?;

    let count = chain.layers.len();
    let out = PyDict::new(py);
    let layers: Vec<i64> = chain.layers.iter().map(|&k| k as i64).collect();
    out.set_item("layers", layers.into_pyarray(py))?;
    out.set_item("depths", matrix(py, count, max_layers - 1, chain.depths)?)?;
    out.set_item("vs", matrix(py, count, max_layers, chain.vs)?)?;
    out.set_item("noise", chain.noise.into_pyarray(py))?;
    let predicted = PyList::empty(py);
    for (curve, values) in space.curves.iter().zip(chain.predicted) {
        predicted.append(matrix(py, count, curve.periods.len(), values)?)?;
    }
    out.set_item("predicted", predicted)?;
    out.set_item("rms", chain.rms.into_pyarray(py))?;
    let (proposed, accepted) = (PyDict::new(py), PyDict::new(py));
    for (i, name) in transdimensional::MOVE_NAMES.iter().enumerate() {
        proposed.set_item(name, chain.proposed[i])?;
        accepted.set_item(name, chain.accepted[i])?;
    }
    out.set_item("proposed", proposed)?;
    out.set_item("accepted", accepted)?;
    out.set_item("swaps", chain.swaps)?;
    let steps = PyDict::new(py);
    steps.set_item("interface", chain.steps.interface)?;
    steps.set_item("vs", chain.steps.vs)?;
    steps.set_item("noise", chain.steps.noise)?;
    steps.set_item("shift", chain.steps.shift)?;
    steps.set_item("stretch", chain.steps.stretch)?;
    out.set_item("steps", steps)?;
    out.set_item("best_depths", chain.best_depths.into_pyarray(py))?;
    out.set_item("best_vs", chain.best_vs.into_pyarray(py))?;
    out.set_item("best_misfit", chain.best_misfit)?;
    Ok(out)
}

/// The fixed layering's chains together by DREAM(ZS) (dream.py's sample), from the curves, the
/// sampled parameters' bounds, where each value goes in the layers, and the settings: each
/// chain's kept states (unit cube, the noise factor last), curves, fit and moves, in a dict.
#[pyfunction]
#[pyo3(signature = (
    curves, low, high, vs_template, thickness_template, vs_from, vs_to, thickness_from,
    thickness_to, vs_index, least_ratio, vp_vs, noise_bounds, n_iterations, n_burnin, save_every,
    n_chains, seed,
))]
#[allow(clippy::too_many_arguments)]
fn dream_chains<'py>(
    py: Python<'py>,
    curves: Vec<CurveArgs<'py>>,
    low: Vec<f64>,
    high: Vec<f64>,
    vs_template: Vec<f64>,
    thickness_template: Vec<f64>,
    vs_from: Vec<usize>,
    vs_to: Vec<usize>,
    thickness_from: Vec<usize>,
    thickness_to: Vec<usize>,
    vs_index: Vec<usize>,
    least_ratio: f64,
    vp_vs: f64,
    noise_bounds: (f64, f64),
    n_iterations: usize,
    n_burnin: usize,
    save_every: usize,
    n_chains: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let names = low.len();
    let layers = vs_template.len();
    let places_fit = |from: &[usize], to: &[usize]| {
        from.len() == to.len() && from.iter().all(|&i| i < names) && to.iter().all(|&i| i < layers)
    };
    if names == 0
        || high.len() != names
        || thickness_template.len() != layers
        || !places_fit(&vs_from, &vs_to)
        || !places_fit(&thickness_from, &thickness_to)
        || vs_index.iter().any(|&i| i >= names)
        || n_chains < 1
        || save_every < 1
    {
        return Err(PyValueError::new_err(
            "the sampled parameters' bounds, their places in the layers or the settings disagree",
        ));
    }
    let problem = dream::Problem {
        curves: curves_from(&curves)?,
        low,
        high,
        vs_template,
        thickness_template,
        vs_from,
        vs_to,
        thickness_from,
        thickness_to,
        vs_index,
        vp_vs,
        least_ratio,
        noise_bounds,
    };
    let settings = dream::Settings {
        n_iterations,
        n_burnin,
        save_every,
        n_chains,
    };
    let sampled = py
        .detach(|| dream::sample(&problem, &settings, seed))
        .map_err(PyValueError::new_err)?;

    let dims = names + 1;
    let out = PyDict::new(py);
    let chains = PyList::empty(py);
    for chain in sampled.chains {
        let count = chain.rms.len();
        let one = PyDict::new(py);
        one.set_item("units", matrix(py, count, dims, chain.units)?)?;
        let predicted = PyList::empty(py);
        for (curve, values) in problem.curves.iter().zip(chain.predicted) {
            predicted.append(matrix(py, count, curve.periods.len(), values)?)?;
        }
        one.set_item("predicted", predicted)?;
        one.set_item("rms", chain.rms.into_pyarray(py))?;
        one.set_item("accepted", chain.accepted)?;
        one.set_item("squared_moves", chain.squared_moves.into_pyarray(py))?;
        chains.append(one)?;
    }
    out.set_item("chains", chains)?;
    out.set_item("best_unit", sampled.best_unit.into_pyarray(py))?;
    let best = PyList::empty(py);
    for values in sampled.best_predicted {
        best.append(values.into_pyarray(py))?;
    }
    out.set_item("best_predicted", best)?;
    out.set_item("crossover", sampled.crossover.to_vec())?;
    Ok(out)
}

/// The phase-shift sum on a regular grid of offsets (x0 + s dx), by Horner's rule: `held` is
/// steps x frequencies, each step's normalized spectra summed; the image's magnitudes,
/// frequencies x velocities.
#[pyfunction]
fn phase_shift_on_grid<'py>(
    py: Python<'py>,
    held: PyReadonlyArray2<'py, Complex64>,
    fs: PyReadonlyArray1<'py, f64>,
    vs: PyReadonlyArray1<'py, f64>,
    x0: f64,
    dx: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (steps, columns) = (held.shape()[0], held.shape()[1]);
    let (h, f, v) = (held.as_slice()?, fs.as_slice()?, vs.as_slice()?);
    if steps == 0 || columns != f.len() {
        return Err(PyValueError::new_err(
            "held must be steps x frequencies, a step at least",
        ));
    }
    let out = py.detach(|| phase_shift::on_grid(h, steps, f, v, x0, dx));
    matrix(py, f.len(), v.len(), out)
}

/// The phase-shift sum at any offsets, directly: `xf` is traces x frequencies, each trace's
/// normalized spectrum; the image's magnitudes, frequencies x velocities.
#[pyfunction]
fn phase_shift_at_offsets<'py>(
    py: Python<'py>,
    xf: PyReadonlyArray2<'py, Complex32>,
    fs: PyReadonlyArray1<'py, f64>,
    vs: PyReadonlyArray1<'py, f64>,
    offsets: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (traces, columns) = (xf.shape()[0], xf.shape()[1]);
    let (x, f, v, o) = (
        xf.as_slice()?,
        fs.as_slice()?,
        vs.as_slice()?,
        offsets.as_slice()?,
    );
    if traces != o.len() || columns != f.len() {
        return Err(PyValueError::new_err(
            "xf must be traces x frequencies, an offset a trace",
        ));
    }
    let out = py.detach(|| phase_shift::at_offsets(x, f, v, o));
    matrix(py, f.len(), v.len(), out)
}

#[pymodule]
fn sigpipe_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("TOLERANCE", surf96::TOLERANCE)?;
    m.add("DispersionError", m.py().get_type::<DispersionError>())?;
    m.add_function(wrap_pyfunction!(rayleigh_phase, m)?)?;
    m.add_function(wrap_pyfunction!(phase_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(period_equation, m)?)?;
    m.add_function(wrap_pyfunction!(transdimensional_chain, m)?)?;
    m.add_function(wrap_pyfunction!(dream_chains, m)?)?;
    m.add_function(wrap_pyfunction!(phase_shift_on_grid, m)?)?;
    m.add_function(wrap_pyfunction!(phase_shift_at_offsets, m)?)?;
    Ok(())
}
