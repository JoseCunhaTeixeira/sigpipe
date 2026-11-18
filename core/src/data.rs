//! What both samplers share (sigpipe's data.py): the picked curves, the forward model on a
//! layered model in m and m/s, and the likelihood with its noise factor.

use crate::surf96::rayleigh_phase;

/// The phase-velocity increment of the roots' bracketing, km/s, as in sigpipe's other forward
/// models.
pub const DC_KM_S: f64 = 0.005;
/// The half-space's thickness given to the forward model, which treats the last layer as a
/// half-space whatever its thickness (data.py's HALF_SPACE_M).
pub const HALF_SPACE_M: f64 = 1_000.0;

/// A picked curve by increasing period, as the samplers take it.
pub struct Curve {
    pub mode: usize,
    pub periods: Vec<f64>,  // s, increasing
    pub observed: Vec<f64>, // m/s
    pub sigma: Vec<f64>,    // m/s, before the noise factor
}

pub fn n_points(curves: &[Curve]) -> usize {
    curves.iter().map(|curve| curve.observed.len()).sum()
}

/// Each curve's phase velocities (m/s, by increasing period) for the layered model (m, m/s, top
/// down; the last layer a half-space), or None where a mode does not exist over its curve's
/// periods. Vp and density follow Vs by `vp_vs` and Gardner's relation.
pub fn phase_velocities(
    curves: &[Curve],
    thickness: &[f64],
    vs: &[f64],
    vp_vs: f64,
) -> Option<Vec<Vec<f64>>> {
    let layers = vs.len();
    let (mut d, mut a, mut b, mut r) = (
        Vec::with_capacity(layers),
        Vec::with_capacity(layers),
        Vec::with_capacity(layers),
        Vec::with_capacity(layers),
    );
    for (&thick, &s) in thickness.iter().zip(vs) {
        let vp = s * vp_vs;
        let rho = 0.32 * vp + 770.0; // kg/m3
        d.push(thick / 1_000.0);
        a.push(vp / 1_000.0);
        b.push(s / 1_000.0);
        r.push(rho / 1_000.0);
    }
    let mut predicted = Vec::with_capacity(curves.len());
    for curve in curves {
        let c = rayleigh_phase(&curve.periods, &d, &a, &b, &r, curve.mode, DC_KM_S).ok()?;
        if !c.iter().all(|&value| value > 0.0) {
            return None;
        }
        predicted.push(c.into_iter().map(|value| value * 1_000.0).collect());
    }
    Some(predicted)
}

/// The sum of the squared residuals, each in its pick's uncertainty.
pub fn misfit(curves: &[Curve], predicted: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for (curve, c) in curves.iter().zip(predicted) {
        for ((observed, sigma), value) in curve.observed.iter().zip(&curve.sigma).zip(c) {
            let residual = (observed - value) / sigma;
            total += residual * residual;
        }
    }
    total
}

/// The residuals' root mean square, m/s.
pub fn rms(curves: &[Curve], predicted: &[Vec<f64>]) -> f64 {
    let mut squares = 0.0;
    for (curve, c) in curves.iter().zip(predicted) {
        for (observed, value) in curve.observed.iter().zip(c) {
            squares += (observed - value) * (observed - value);
        }
    }
    (squares / n_points(curves) as f64).sqrt()
}

/// The likelihood's logarithm with the picks' uncertainties scaled by exp(`log_noise`) (its
/// constant left out).
pub fn log_likelihood(misfit_value: f64, log_noise: f64, n_points: usize) -> f64 {
    -(n_points as f64) * log_noise - misfit_value / (2.0 * (2.0 * log_noise).exp())
}

/// Whether no layer's Vs falls below `least_ratio` of the one above it.
pub fn allowed(vs: &[f64], least_ratio: f64) -> bool {
    least_ratio <= 0.0 || vs.windows(2).all(|pair| pair[1] >= least_ratio * pair[0])
}
