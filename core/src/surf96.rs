// Derived from Keurfon Luu's numba port of surf96 (Computer Programs in Seismology,
// R. B. Herrmann), whose license follows.
//
// BSD 3-Clause License
//
// Copyright (c) 2020, Keurfon Luu
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The notice of Computer Programs in Seismology, which that port carries:
//
//     COMPUTER PROGRAMS IN SEISMOLOGY
//     VOLUME IV
//
//     COPYRIGHT 1986, 1991
//     D. R. Russell, R. B. Herrmann
//     Department of Earth and Atmospheric Sciences
//     Saint Louis University
//     221 North Grand Boulevard
//     St. Louis, Missouri 63103
//     U. S. A.

//! Phase velocities of Rayleigh and Love waves in a layered model: surf96's period equations
//! (Dunkin's compound matrices for Rayleigh waves, Thomson-Haskell propagation for Love waves)
//! and its root search.
//!
//! Each root is bracketed by steps of `dc` from the root at the period before, then refined by
//! Illinois' regula falsi (the secant through the bracket's ends, the end kept twice in a row
//! weighed half, a halving where the secant leaves the bracket) until the bracket is TOLERANCE
//! of the root wide. Where a bracketing step holds several roots, as it may at high frequencies
//! over thick soft layers, where modes lie a fraction of a m/s apart, the search returns one of
//! them, not necessarily the mode asked. The Love waves' equation, a product of propagators
//! normalized layer by layer, may also change sign across a pole, which the search then takes
//! for a root: at higher modes of strongly layered models.

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
/// A root is refined until its bracket is this share of it wide.
pub const TOLERANCE: f64 = 1.0e-6;

/// The fundamental mode was not found over the periods asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoFundamentalMode;

/// The surface wave whose period equation is solved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wave {
    Rayleigh,
    Love,
}

/// A layered model in km, km/s and g/cm3, top down, the last layer a half-space; `llw` is 0
/// when the top layer is water (no S wave), -1 otherwise.
struct Model<'a> {
    d: &'a [f64],
    a: &'a [f64],
    b: &'a [f64],
    rho: &'a [f64],
    llw: isize,
}

impl<'a> Model<'a> {
    fn new(d: &'a [f64], a: &'a [f64], b: &'a [f64], rho: &'a [f64]) -> Self {
        Model {
            d,
            a,
            b,
            rho,
            llw: if b[0] <= 0.0 { 0 } else { -1 },
        }
    }
}

/// The sign of `x`: 0 at 0, NaN for NaN.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

/// Dunkin's vector normalized by its largest component.
fn normc(ee: &mut [f64; 5]) {
    let mut t1 = 0.0_f64;
    for value in ee.iter() {
        t1 = t1.max(value.abs());
    }
    if t1 < 1.0e-40 {
        t1 = 1.0;
    }
    for value in ee.iter_mut() {
        *value /= t1;
    }
}

/// A layer's eigenfunction products, for the compound matrices.
struct Products {
    w: f64,
    cosp: f64,
    a0: f64,
    cpcq: f64,
    cpy: f64,
    cpz: f64,
    cqw: f64,
    cqx: f64,
    xy: f64,
    xz: f64,
    wy: f64,
    wz: f64,
}

/// The layer's P- and S-wave eigenfunctions (cosP, sinP, cosQ, sinQ, as the phase velocity is
/// below, at or above each wave's velocity) and their products.
#[allow(clippy::too_many_arguments)]
fn var(p: f64, q: f64, ra: f64, rb: f64, wvno: f64, xka: f64, xkb: f64, dpth: f64) -> Products {
    let mut pex = 0.0;
    let (cosp, w, x);
    if wvno < xka {
        let sinp = p.sin();
        w = sinp / ra;
        x = -ra * sinp;
        cosp = p.cos();
    } else if wvno == xka {
        cosp = 1.0;
        w = dpth;
        x = 0.0;
    } else {
        pex = p;
        let fac = if p < 16.0 { (-2.0 * p).exp() } else { 0.0 };
        cosp = (1.0 + fac) * 0.5;
        let sinp = (1.0 - fac) * 0.5;
        w = sinp / ra;
        x = ra * sinp;
    }

    let mut sex = 0.0;
    let (cosq, y, z);
    if wvno < xkb {
        let sinq = q.sin();
        y = sinq / rb;
        z = -rb * sinq;
        cosq = q.cos();
    } else if wvno == xkb {
        cosq = 1.0;
        y = dpth;
        z = 0.0;
    } else {
        sex = q;
        let fac = if q < 16.0 { (-2.0 * q).exp() } else { 0.0 };
        cosq = (1.0 + fac) * 0.5;
        let sinq = (1.0 - fac) * 0.5;
        y = sinq / rb;
        z = rb * sinq;
    }

    let exa = pex + sex;
    Products {
        w,
        cosp,
        a0: if exa < 60.0 { (-exa).exp() } else { 0.0 },
        cpcq: cosp * cosq,
        cpy: cosp * y,
        cpz: cosp * z,
        cqw: cosq * w,
        cqx: cosq * x,
        xy: x * y,
        xz: x * z,
        wy: w * y,
        wz: w * z,
    }
}

/// Dunkin's compound matrix of a layer.
fn dnka(ca: &mut [[f64; 5]; 5], wvno2: f64, gam: f64, gammk: f64, rho: f64, v: &Products) {
    let gamm1 = gam - 1.0;
    let twgm1 = gam + gamm1;
    let gmgmk = gam * gammk;
    let gmgm1 = gam * gamm1;
    let gm1sq = gamm1 * gamm1;

    let rho2 = rho * rho;
    let a0pq = v.a0 - v.cpcq;
    let t = -2.0 * wvno2;

    ca[0][0] = v.cpcq - 2.0 * gmgm1 * a0pq - gmgmk * v.xz - wvno2 * gm1sq * v.wy;
    ca[0][1] = (wvno2 * v.cpy - v.cqx) / rho;
    ca[0][2] = -(twgm1 * a0pq + gammk * v.xz + wvno2 * gamm1 * v.wy) / rho;
    ca[0][3] = (v.cpz - wvno2 * v.cqw) / rho;
    ca[0][4] = -(2.0 * wvno2 * a0pq + v.xz + wvno2 * wvno2 * v.wy) / rho2;

    ca[1][0] = (gmgmk * v.cpz - gm1sq * v.cqw) * rho;
    ca[1][1] = v.cpcq;
    ca[1][2] = gammk * v.cpz - gamm1 * v.cqw;
    ca[1][3] = -v.wz;
    ca[1][4] = ca[0][3];

    ca[3][0] = (gm1sq * v.cpy - gmgmk * v.cqx) * rho;
    ca[3][1] = -v.xy;
    ca[3][2] = gamm1 * v.cpy - gammk * v.cqx;
    ca[3][3] = ca[1][1];
    ca[3][4] = ca[0][1];

    ca[4][0] = -(2.0 * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * v.xz + gm1sq * gm1sq * v.wy) * rho2;
    ca[4][1] = ca[3][0];
    ca[4][2] =
        -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * v.xz + gamm1 * gm1sq * v.wy) * rho;
    ca[4][3] = ca[1][0];
    ca[4][4] = ca[0][0];

    ca[2][0] = t * ca[4][2];
    ca[2][1] = t * ca[3][2];
    ca[2][2] = v.a0 + 2.0 * (v.cpcq - ca[0][0]);
    ca[2][3] = t * ca[1][2];
    ca[2][4] = t * ca[0][2];
}

/// The Rayleigh-wave period equation at wavenumber `wvno`: Dunkin's matrices from the
/// half-space up.
fn dltar4(wvno: f64, omega: f64, m: &Model) -> f64 {
    let n = m.d.len();
    let omega = omega.max(1.0e-4);
    let wvno2 = wvno * wvno;
    let mut xka = omega / m.a[n - 1];
    let mut xkb = omega / m.b[n - 1];
    let mut ra = ((wvno + xka) * (wvno - xka).abs()).sqrt();
    let mut rb = ((wvno + xkb) * (wvno - xkb).abs()).sqrt();
    let t = m.b[n - 1] / omega;

    // The half-space's E matrix.
    let gammk = 2.0 * t * t;
    let gam = gammk * wvno2;
    let gamm1 = gam - 1.0;
    let rho1 = m.rho[n - 1];
    let mut e = [
        rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb),
        -rho1 * ra,
        rho1 * (gamm1 - gammk * ra * rb),
        rho1 * rb,
        wvno2 - ra * rb,
    ];
    let mut ca = [[0.0_f64; 5]; 5];

    let mut q = 0.0;
    let mut layer = n as isize - 2;
    while layer > m.llw {
        let i = layer as usize;
        xka = omega / m.a[i];
        xkb = omega / m.b[i];
        let t = m.b[i] / omega;
        let gammk = 2.0 * t * t;
        let gam = gammk * wvno2;
        ra = ((wvno + xka) * (wvno - xka).abs()).sqrt();
        rb = ((wvno + xkb) * (wvno - xkb).abs()).sqrt();

        let dpth = m.d[i];
        let p = ra * dpth;
        q = rb * dpth;
        let products = var(p, q, ra, rb, wvno, xka, xkb, dpth);
        dnka(&mut ca, wvno2, gam, gammk, m.rho[i], &products);

        let mut ee = [0.0_f64; 5];
        for (column, value) in ee.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, e_row) in e.iter().enumerate() {
                sum += e_row * ca[row][column];
            }
            *value = sum;
        }
        normc(&mut ee);
        e = ee;
        layer -= 1;
    }

    if m.llw == 0 {
        xka = omega / m.a[0];
        ra = ((wvno + xka) * (wvno - xka).abs()).sqrt();
        let dpth = m.d[0];
        let p = ra * dpth;
        let products = var(p, q, ra, 1.0e-5, wvno, xka, xkb, dpth);
        products.cosp * e[0] - m.rho[0] * products.w * e[1]
    } else {
        e[0]
    }
}

/// The Love-wave period equation at wavenumber `wvno`: Thomson-Haskell propagation from the
/// half-space up.
fn dltar1(wvno: f64, omega: f64, m: &Model) -> f64 {
    let n = m.d.len();
    let beta = m.b[n - 1];
    let xkb = omega / beta;
    let rb = ((wvno + xkb) * (wvno - xkb).abs()).sqrt();
    let mut e1 = m.rho[n - 1] * rb;
    let mut e2 = 1.0 / (beta * beta);

    let mut layer = n as isize - 2;
    while layer > m.llw {
        let i = layer as usize;
        let beta = m.b[i];
        let xmu = m.rho[i] * beta * beta;
        let xkb = omega / beta;
        let rb = ((wvno + xkb) * (wvno - xkb).abs()).sqrt();
        let q = m.d[i] * rb;

        let (cosq, y, z);
        if wvno < xkb {
            let sinq = q.sin();
            y = sinq / rb;
            z = -rb * sinq;
            cosq = q.cos();
        } else if wvno == xkb {
            cosq = 1.0;
            y = m.d[i];
            z = 0.0;
        } else {
            let fac = if q < 16.0 { (-2.0 * q).exp() } else { 0.0 };
            cosq = (1.0 + fac) * 0.5;
            let sinq = (1.0 - fac) * 0.5;
            y = sinq / rb;
            z = rb * sinq;
        }

        let e10 = e1 * cosq + e2 * xmu * z;
        let e20 = e1 * y / xmu + e2 * cosq;
        let mut norm = e10.abs().max(e20.abs());
        if norm < 1.0e-40 {
            norm = 1.0;
        }
        e1 = e10 / norm;
        e2 = e20 / norm;
        layer -= 1;
    }
    e1
}

/// The period equation of `wave` at wavenumber `wvno` and angular frequency `omega`.
fn dltar(wave: Wave, wvno: f64, omega: f64, m: &Model) -> f64 {
    match wave {
        Wave::Rayleigh => dltar4(wvno, omega, m),
        Wave::Love => dltar1(wvno, omega, m),
    }
}

/// The root bracketed by `c1` and `c2` (the period equation `del1` and `del2` there), by
/// Illinois' regula falsi, until the bracket is TOLERANCE of its end wide.
#[allow(clippy::too_many_arguments)]
fn refine(
    wave: Wave,
    omega: f64,
    mut c1: f64,
    mut c2: f64,
    mut del1: f64,
    mut del2: f64,
    m: &Model,
) -> f64 {
    let mut c3 = 0.5 * (c1 + c2);
    let mut side = 0_i8;
    for _ in 0..100 {
        if del2 != del1 {
            c3 = (c1 * del2 - c2 * del1) / (del2 - del1);
        }
        let (low, high) = (c1.min(c2), c1.max(c2));
        if !(low < c3 && c3 < high) {
            c3 = 0.5 * (c1 + c2);
        }
        let del3 = dltar(wave, omega / c3, omega, m);
        if del3 == 0.0 {
            break;
        }
        if sign(del3) * sign(del1) < 0.0 {
            c2 = c3;
            del2 = del3;
            if side == -1 {
                del1 *= 0.5;
            }
            side = -1;
        } else {
            c1 = c3;
            del1 = del3;
            if side == 1 {
                del2 *= 0.5;
            }
            side = 1;
        }
        if (c1 - c2).abs() <= TOLERANCE * c1 {
            break;
        }
    }
    c3
}

/// The root at period `t1`, bracketed from `c1` by steps of `dc` (never below `clow`), then
/// refined; with the period equation at the first period's start, and whether the search left
/// the model's velocities.
#[allow(clippy::too_many_arguments)]
fn getsol(
    wave: Wave,
    t1: f64,
    mut c1: f64,
    clow: f64,
    dc: f64,
    cm: f64,
    betmx: f64,
    ifirst: bool,
    mut del1st: f64,
    m: &Model,
) -> (f64, f64, bool) {
    let omega = TWO_PI / t1;
    let mut del1 = dltar(wave, omega / c1, omega, m);
    if ifirst {
        del1st = del1;
    }
    // The search goes down where the period equation's sign differs from the first period's.
    let mut idir = if !ifirst && sign(del1st) * sign(del1) < 0.0 {
        -1.0
    } else {
        1.0
    };
    loop {
        let c2 = c1 + idir * dc;
        if c2 <= clow {
            idir = 1.0;
            c1 = clow;
            continue;
        }
        let del2 = dltar(wave, omega / c2, omega, m);
        if sign(del1) != sign(del2) {
            let root = refine(wave, omega, c1, c2, del1, del2, m);
            return (root, del1st, root > betmx);
        }
        c1 = c2;
        del1 = del2;
        if c1 < cm || c1 >= betmx + dc {
            return (c1, del1st, true);
        }
    }
}

/// A half-space's Rayleigh velocity (P- and S-wave velocities `a` and `b`), the search's start.
fn gtsolh(a: f64, b: f64) -> f64 {
    let mut c = 0.95 * b;
    for _ in 0..5 {
        let gamma = b / a;
        let kappa = c / b;
        let k2 = kappa * kappa;
        let gk2 = (gamma * kappa) * (gamma * kappa);
        let fac1 = (1.0 - gk2).sqrt();
        let fac2 = (1.0 - k2).sqrt();
        let fr = (2.0 - k2) * (2.0 - k2) - 4.0 * fac1 * fac2;
        let mut frp = -4.0 * (2.0 - k2) * kappa;
        frp += 4.0 * fac2 * gamma * gamma * kappa / fac1;
        frp += 4.0 * fac1 * kappa / fac2;
        frp /= b;
        c -= fr / frp;
    }
    c
}

/// The phase velocities (km/s) of `wave`'s mode `mode` (0: the fundamental) at periods `t`
/// (s, increasing), of the layers' thicknesses `d` (km, the last one's ignored: a half-space),
/// P- and S-wave velocities `a` and `b` (km/s) and densities `rho` (g/cm3), top down; 0 at the
/// periods where a higher mode does not exist. The roots are bracketed by steps of `dc` (km/s).
/// Err where the fundamental mode is not found.
#[allow(clippy::too_many_arguments)]
pub fn phase_velocities(
    t: &[f64],
    d: &[f64],
    a: &[f64],
    b: &[f64],
    rho: &[f64],
    mode: usize,
    wave: Wave,
    dc: f64,
) -> Result<Vec<f64>, NoFundamentalMode> {
    let m = Model::new(d, a, b, rho);
    let kmax = t.len();
    let mut c = vec![0.0; kmax];
    let mut cg = vec![0.0; kmax];

    // The extremal velocities, which bound the search.
    let mut betmx = -1.0e20_f64;
    let mut betmn = 1.0e20_f64;
    let mut jmn = 0;
    let mut jsol = false;
    for i in 0..d.len() {
        if b[i] > 0.01 && b[i] < betmn {
            betmn = b[i];
            jmn = i;
            jsol = false;
        } else if b[i] < 0.01 && a[i] < betmn {
            betmn = a[i];
            jmn = i;
            jsol = true;
        }
        betmx = betmx.max(b[i]);
    }

    // The search starts a little below the slowest layer's half-space solution.
    let cc = 0.9 * if jsol { betmn } else { gtsolh(a[jmn], b[jmn]) };
    let mut c1 = cc;
    let cm = cc;

    let one = 1.0e-2;
    let onea = 1.5;
    for iq in 0..=mode {
        let mut del1st = 0.0;
        for k in 0..kmax {
            // Each mode starts above the one below it, each period near the period before.
            let ifirst = k == 0;
            let clow;
            if ifirst && iq == 0 {
                clow = cc;
                c1 = cc;
            } else if ifirst {
                clow = c1;
                c1 = c[0] + one * dc;
            } else if iq > 0 {
                clow = c[k] + one * dc;
                c1 = c[k - 1].max(clow);
            } else {
                clow = cm;
                c1 = c[k - 1] - onea * dc;
            }

            let (root, first, iret) =
                getsol(wave, t[k], c1, clow, dc, cm, betmx, ifirst, del1st, &m);
            c1 = root;
            del1st = first;
            if iret {
                if iq == 0 {
                    return Err(NoFundamentalMode);
                }
                for value in cg.iter_mut().skip(k) {
                    *value = 0.0;
                }
                if iq == mode {
                    return Ok(cg);
                }
                c1 = 0.0;
                break;
            }
            c[k] = c1;
            cg[k] = c1;
            c1 = 0.0;
        }
    }
    Ok(cg)
}

/// The Rayleigh-wave phase velocities of `phase_velocities`, the samplers' forward model.
pub fn rayleigh_phase(
    t: &[f64],
    d: &[f64],
    a: &[f64],
    b: &[f64],
    rho: &[f64],
    mode: usize,
    dc: f64,
) -> Result<Vec<f64>, NoFundamentalMode> {
    phase_velocities(t, d, a, b, rho, mode, Wave::Rayleigh, dc)
}

/// `wave`'s period equation at phase velocity `c` (km/s) and period `t` (s): a check of the
/// roots found.
pub fn period_equation(
    wave: Wave,
    t: f64,
    c: f64,
    d: &[f64],
    a: &[f64],
    b: &[f64],
    rho: &[f64],
) -> f64 {
    let omega = TWO_PI / t;
    dltar(wave, omega / c, omega, &Model::new(d, a, b, rho))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layered() -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        let d = [0.003, 0.006, 1.0];
        let b = [0.18, 0.32, 0.55];
        let a = b.map(|v| v * 1.77);
        let rho = a.map(|v| (0.32 * v * 1000.0 + 770.0) / 1000.0);
        (d, a, b, rho)
    }

    #[test]
    fn a_poisson_half_space_has_its_rayleigh_wave_at_every_period() {
        // Vp = sqrt(3) Vs: the Rayleigh wave at sqrt(2 - 2 / sqrt(3)) of Vs.
        let vs = 0.3;
        let periods: Vec<f64> = (1..=30).map(|k| 1.0 / (80.0 - 2.5 * k as f64)).collect();
        let c = rayleigh_phase(
            &periods,
            &[1.0],
            &[3.0_f64.sqrt() * vs],
            &[vs],
            &[2.0],
            0,
            0.005,
        )
        .unwrap();
        let expected = (2.0 - 2.0 / 3.0_f64.sqrt()).sqrt() * vs;
        for value in c {
            assert!(
                (value - expected).abs() <= 2.0 * TOLERANCE * expected,
                "{value} {expected}"
            );
        }
    }

    #[test]
    fn every_velocity_found_is_a_root() {
        let (d, a, b, rho) = layered();
        let periods: Vec<f64> = (0..20).map(|k| 1.0 / (60.0 - 2.0 * k as f64)).collect();
        for wave in [Wave::Rayleigh, Wave::Love] {
            for mode in 0..2 {
                let c = phase_velocities(&periods, &d, &a, &b, &rho, mode, wave, 0.005).unwrap();
                for (&t, &v) in periods.iter().zip(&c) {
                    if v > 0.0 {
                        let below =
                            period_equation(wave, t, v * (1.0 - 4.0 * TOLERANCE), &d, &a, &b, &rho);
                        let above =
                            period_equation(wave, t, v * (1.0 + 4.0 * TOLERANCE), &d, &a, &b, &rho);
                        assert!(sign(below) != sign(above), "{wave:?} M{mode} at {t} s: {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn love_waves_lie_between_the_top_layer_and_the_half_space() {
        // A soft layer over a stiff half-space: the fundamental Love mode starts near the
        // layer's Vs at short periods and rises toward the half-space's at long ones.
        let (d, a, b, rho) = ([0.005, 1.0], [0.354, 0.885], [0.2, 0.5], [1.88, 2.1]);
        let periods: Vec<f64> = (0..25).map(|k| 1.0 / (60.0 - 2.2 * k as f64)).collect();
        let c = phase_velocities(&periods, &d, &a, &b, &rho, 0, Wave::Love, 0.005).unwrap();
        assert!(c.iter().all(|&v| v > 0.2 && v < 0.5), "{c:?}");
        assert!(c.windows(2).all(|pair| pair[1] >= pair[0]), "{c:?}");
    }
}
