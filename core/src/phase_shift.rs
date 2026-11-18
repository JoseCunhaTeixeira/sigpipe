//! The phase-shift transform's sum (Park et al. 1998), its delay and sum over the traces at each
//! frequency and velocity: sigpipe's phase_shift.py normalizes the spectra and calls these.

use numpy::{Complex32, Complex64};

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// On a regular grid of offsets, by Horner's rule over its steps: |sum over the steps s of
/// held[s, f] exp(i k (x0 + s dx))|, k = 2 pi f / v, for each frequency f and velocity v (rows
/// and columns of the result). `held` is steps x frequencies, row-major, each step's traces
/// summed (an empty step holds none): two exponentials a frequency and velocity, where the
/// direct sum takes one a trace.
pub fn on_grid(
    held: &[Complex64],
    steps: usize,
    fs: &[f64],
    vs: &[f64],
    x0: f64,
    dx: f64,
) -> Vec<f64> {
    let (nf, nv) = (fs.len(), vs.len());
    let mut out = vec![0.0; nf * nv];
    let mut column = vec![Complex64::new(0.0, 0.0); steps];
    for (f, &frequency) in fs.iter().enumerate() {
        for (s, value) in column.iter_mut().enumerate() {
            *value = held[s * nf + f];
        }
        let two_pi_f = TWO_PI * frequency;
        for (j, &velocity) in vs.iter().enumerate() {
            let per_metre = two_pi_f / (velocity + 1e-12);
            let step = Complex64::from_polar(1.0, per_metre * dx);
            let mut sum = column[steps - 1];
            for value in column[..steps - 1].iter().rev() {
                sum = sum * step + value;
            }
            sum *= Complex64::from_polar(1.0, per_metre * x0);
            out[f * nv + j] = sum.norm();
        }
    }
    out
}

/// At any offsets, directly: |sum over the traces x of xf[x, f] exp(i 2 pi offset_x f / v)|
/// for each frequency f and velocity v. `xf` is traces x frequencies, row-major.
pub fn at_offsets(xf: &[Complex32], fs: &[f64], vs: &[f64], offsets: &[f64]) -> Vec<f64> {
    let (nf, nv, nx) = (fs.len(), vs.len(), offsets.len());
    let mut out = vec![0.0; nf * nv];
    let mut column = vec![Complex64::new(0.0, 0.0); nx];
    for (f, &frequency) in fs.iter().enumerate() {
        for (x, value) in column.iter_mut().enumerate() {
            let spectrum = xf[x * nf + f];
            *value = Complex64::new(spectrum.re as f64, spectrum.im as f64);
        }
        for (j, &velocity) in vs.iter().enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);
            for (value, &offset) in column.iter().zip(offsets) {
                sum += value
                    * Complex64::from_polar(1.0, TWO_PI * offset * frequency / (velocity + 1e-12));
            }
            out[f * nv + j] = sum.norm();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_sum_is_the_direct_sum() {
        // Six traces on a grid of 1.5 m from 3 m, one step empty, one holding two traces.
        let steps_of = [0_usize, 1, 2, 2, 4, 5];
        let offsets: Vec<f64> = steps_of.iter().map(|&s| 3.0 + 1.5 * s as f64).collect();
        let fs = [5.0, 12.5, 31.0];
        let vs = [120.0, 300.0, 850.0];
        let xf: Vec<Complex32> = (0..steps_of.len() * fs.len())
            .map(|k| Complex32::from_polar(1.0, 0.37 * k as f32))
            .collect();
        let mut held = vec![Complex64::new(0.0, 0.0); 6 * fs.len()];
        for (x, &s) in steps_of.iter().enumerate() {
            for f in 0..fs.len() {
                let value = xf[x * fs.len() + f];
                held[s * fs.len() + f] += Complex64::new(value.re as f64, value.im as f64);
            }
        }
        let grid = on_grid(&held, 6, &fs, &vs, 3.0, 1.5);
        let direct = at_offsets(&xf, &fs, &vs, &offsets);
        for (a, b) in grid.iter().zip(&direct) {
            assert!((a - b).abs() <= 1e-9 * b.max(1.0), "{a} {b}");
        }
    }
}
