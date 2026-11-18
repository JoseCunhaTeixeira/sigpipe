//! Layered Vs models whose number of layers the data choose: reversible-jump Markov chains with
//! parallel tempering. sigpipe's transdimensional.py documents the model, the priors and the
//! moves.

use crate::data::{
    Curve, HALF_SPACE_M, allowed, log_likelihood, misfit, n_points, phase_velocities, rms,
};
use crate::rng::Random;

/// The new layer's Vs around the one it splits from, on the logarithm.
const BIRTH_SPREAD: f64 = 0.25;
/// The local moves' acceptance aimed at during the burn-in.
const TARGET: f64 = 0.3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Move {
    Birth,
    Death,
    Interface,
    Relocate,
    Vs,
    Noise,
    Shift,
    Stretch,
}

/// Each move's probability: birth, death, an interface's step, an interface anywhere between its
/// neighbours, a Vs, the noise factor, every Vs, every depth.
const MOVES: [Move; 8] = [
    Move::Birth,
    Move::Death,
    Move::Interface,
    Move::Relocate,
    Move::Vs,
    Move::Noise,
    Move::Shift,
    Move::Stretch,
];
const WEIGHTS: [f64; 8] = [0.12, 0.12, 0.15, 0.06, 0.29, 0.08, 0.09, 0.09];

/// The data and the priors.
pub struct Space {
    pub curves: Vec<Curve>,
    pub vs_bounds: (f64, f64),    // m/s
    pub depth_bounds: (f64, f64), // m: the shallowest and deepest interface
    pub max_layers: usize,        // the half-space included
    pub least_ratio: f64,         // a layer's Vs over the one above it, at least (0: any)
    pub vp_vs: f64,
    pub noise_bounds: (f64, f64), // the noise factor's prior (data.py's NOISE_BOUNDS)
    pub thinnest: f64,            // a layer's least thickness, a share of its top's depth
}

pub struct Settings {
    pub n_iterations: usize,
    pub n_burnin: usize,
    pub save_every: usize,
    pub temperatures: usize, // copies of the chain, the first at the posterior
    pub hottest: f64,        // the hottest copy's temperature
}

/// The steps of the moves that adapt them, in the logarithm of the values moved.
#[derive(Clone, Copy)]
pub struct Steps {
    pub interface: f64,
    pub vs: f64,
    pub noise: f64,
    pub shift: f64,
    pub stretch: f64,
}

impl Steps {
    fn of(&mut self, step: Move) -> Option<&mut f64> {
        match step {
            Move::Interface => Some(&mut self.interface),
            Move::Vs => Some(&mut self.vs),
            Move::Noise => Some(&mut self.noise),
            Move::Shift => Some(&mut self.shift),
            Move::Stretch => Some(&mut self.stretch),
            Move::Birth | Move::Death | Move::Relocate => None,
        }
    }
}

/// One tempered copy: its model (log depths, log Vs, log noise), fit and steps.
struct Copy {
    log_depths: Vec<f64>,
    log_vs: Vec<f64>,
    log_noise: f64,
    misfit: f64,
    likelihood: f64,
    predicted: Vec<Vec<f64>>,
    steps: Steps,
}

/// One chain's kept models (padded with NaN beyond their layers) and how it ran.
pub struct Chain {
    pub layers: Vec<usize>,
    pub depths: Vec<f64>, // kept x (max_layers - 1), m
    pub vs: Vec<f64>,     // kept x max_layers, m/s
    pub noise: Vec<f64>,
    pub predicted: Vec<Vec<f64>>, // per curve: kept x points, by increasing period
    pub rms: Vec<f64>,
    pub proposed: [usize; 8], // each move's, of the first copy
    pub accepted: [usize; 8],
    pub swaps: (usize, usize), // the exchanges accepted, and tried
    pub steps: Steps,          // the first copy's, after the burn-in
    pub best_depths: Vec<f64>, // the best model visited (least misfit), at any temperature
    pub best_vs: Vec<f64>,
    pub best_misfit: f64,
}

/// The move names, in the order of Chain's counts.
pub const MOVE_NAMES: [&str; 8] = [
    "birth",
    "death",
    "interface",
    "relocate",
    "vs",
    "noise",
    "shift",
    "stretch",
];

struct Bounds {
    w_low: f64,
    w_high: f64,
    u_low: f64,
    u_high: f64,
    gap: f64,
    n_points: usize,
}

/// A candidate's predicted curves, misfit and log-likelihood.
type Evaluated = (Vec<Vec<f64>>, f64, f64);
/// A kept model: its interfaces' depths, its layers' Vs, its noise factor and its curves.
type Kept = (Vec<f64>, Vec<f64>, f64, Vec<Vec<f64>>);

/// The candidate's predicted curves, misfit and likelihood, or None outside the priors.
fn evaluate(
    space: &Space,
    bounds: &Bounds,
    log_depths: &[f64],
    log_vs: &[f64],
    log_noise: f64,
) -> Option<Evaluated> {
    if let (Some(&first), Some(&last)) = (log_depths.first(), log_depths.last())
        && (first < bounds.u_low
            || last > bounds.u_high
            || log_depths
                .windows(2)
                .any(|pair| pair[1] - pair[0] < bounds.gap))
    {
        return None;
    }
    if log_vs
        .iter()
        .any(|&w| w < bounds.w_low || w > bounds.w_high)
    {
        return None;
    }
    let vs: Vec<f64> = log_vs.iter().map(|w| w.exp()).collect();
    if !allowed(&vs, space.least_ratio) {
        return None;
    }
    let mut thickness = Vec::with_capacity(vs.len());
    let mut top = 0.0;
    for log_depth in log_depths {
        let depth = log_depth.exp();
        thickness.push(depth - top);
        top = depth;
    }
    thickness.push(HALF_SPACE_M);
    let predicted = phase_velocities(&space.curves, &thickness, &vs, space.vp_vs)?;
    let value = misfit(&space.curves, &predicted);
    let likelihood = log_likelihood(value, log_noise, bounds.n_points);
    Some((predicted, value, likelihood))
}

/// `x` folded back within [low, high], as by mirrors at its ends.
fn reflect(x: f64, low: f64, high: f64) -> f64 {
    let span = high - low;
    let y = (x - low).rem_euclid(2.0 * span);
    low + if y <= span { y } else { 2.0 * span - y }
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

/// A model of the priors with few layers, its Vs increasing, whose modes exist.
fn start(space: &Space, bounds: &Bounds, rng: &mut Random) -> Result<Copy, String> {
    for _ in 0..10_000 {
        let k = 1 + rng.integer(space.max_layers.min(3));
        let log_depths = sorted(
            (1..k)
                .map(|_| rng.uniform(bounds.u_low, bounds.u_high))
                .collect(),
        );
        let log_vs = sorted(
            (0..k)
                .map(|_| rng.uniform(bounds.w_low, bounds.w_high))
                .collect(),
        );
        let log_noise = space.noise_bounds.1.ln();
        if let Some((predicted, value, likelihood)) =
            evaluate(space, bounds, &log_depths, &log_vs, log_noise)
        {
            return Ok(Copy {
                log_depths,
                log_vs,
                log_noise,
                misfit: value,
                likelihood,
                predicted,
                steps: Steps {
                    interface: 0.1,
                    vs: 0.05,
                    noise: 0.1,
                    shift: 0.02,
                    stretch: 0.02,
                },
            });
        }
    }
    Err(
        "No model of the priors has the picked modes over their frequencies, within the Vs ratios \
         allowed: widen the Vs bounds."
            .to_string(),
    )
}

/// One chain and its hotter copies, from random models of the priors.
pub fn run_chain(space: &Space, settings: &Settings, seed: u64) -> Result<Chain, String> {
    let mut rng = Random::new(seed);
    let bounds = Bounds {
        w_low: space.vs_bounds.0.ln(),
        w_high: space.vs_bounds.1.ln(),
        u_low: space.depth_bounds.0.ln(),
        u_high: space.depth_bounds.1.ln(),
        gap: (1.0 + space.thinnest).ln(),
        n_points: n_points(&space.curves),
    };
    let w_span = bounds.w_high - bounds.w_low;
    let (s_low, s_high) = (space.noise_bounds.0.ln(), space.noise_bounds.1.ln());
    let k_count = settings.temperatures.max(1);
    let betas: Vec<f64> = (0..k_count)
        .map(|t| {
            if k_count > 1 {
                1.0 / settings.hottest.powf(t as f64 / (k_count - 1) as f64)
            } else {
                1.0
            }
        })
        .collect();
    let total: f64 = WEIGHTS.iter().sum();
    let mut cumulative = [0.0; 8];
    let mut running = 0.0;
    for (slot, weight) in cumulative.iter_mut().zip(WEIGHTS) {
        running += weight / total;
        *slot = running;
    }
    let log_2pi = (2.0 * std::f64::consts::PI).ln();

    let mut copies: Vec<Copy> = Vec::with_capacity(k_count);
    for _ in 0..k_count {
        copies.push(start(space, &bounds, &mut rng)?);
    }
    let mut best = 0;
    for (t, copy) in copies.iter().enumerate() {
        if copy.misfit < copies[best].misfit {
            best = t;
        }
    }
    let mut best_misfit = copies[best].misfit;
    let mut best_depths: Vec<f64> = copies[best].log_depths.iter().map(|x| x.exp()).collect();
    let mut best_vs: Vec<f64> = copies[best].log_vs.iter().map(|w| w.exp()).collect();

    let mut proposed = [0_usize; 8];
    let mut accepted = [0_usize; 8];
    let mut swaps = (0_usize, 0_usize);
    let mut kept: Vec<Kept> = Vec::new();

    for iteration in 1..=settings.n_iterations {
        let burning = iteration <= settings.n_burnin;
        let gain = 1.0 / (1.0 + iteration as f64 / 100.0).powf(0.6);
        for t in 0..k_count {
            let u = rng.random();
            let index = cumulative
                .iter()
                .position(|&c| c >= u)
                .unwrap_or(MOVES.len() - 1);
            let step = MOVES[index];
            let copy = &copies[t];
            let k = copy.log_vs.len();
            let mut log_depths = copy.log_depths.clone();
            let mut log_vs = copy.log_vs.clone();
            let mut log_noise = copy.log_noise;
            let mut extra = 0.0; // the log prior and proposal ratio
            match step {
                Move::Birth => {
                    if k >= space.max_layers {
                        continue;
                    }
                    let depth = rng.uniform(bounds.u_low, bounds.u_high);
                    let j = log_depths.partition_point(|&x| x < depth);
                    let eps = rng.normal();
                    let new_vs = log_vs[j] + BIRTH_SPREAD * eps;
                    log_depths.insert(j, depth);
                    log_vs.insert(j + 1, new_vs);
                    extra = BIRTH_SPREAD.ln() - w_span.ln() + 0.5 * log_2pi + 0.5 * eps * eps;
                }
                Move::Death => {
                    if k <= 1 {
                        continue;
                    }
                    let i = rng.integer(k - 1);
                    let eps = (log_vs[i + 1] - log_vs[i]) / BIRTH_SPREAD;
                    log_depths.remove(i);
                    log_vs.remove(i + 1);
                    extra = -BIRTH_SPREAD.ln() + w_span.ln() - 0.5 * log_2pi - 0.5 * eps * eps;
                }
                Move::Interface => {
                    if k <= 1 {
                        continue;
                    }
                    let i = rng.integer(k - 1);
                    log_depths[i] += copy.steps.interface * rng.normal();
                }
                Move::Relocate => {
                    // Anywhere between its neighbours, whatever its place: an interface whose
                    // depth the data leave between two places crosses to the other at once.
                    if k <= 1 {
                        continue;
                    }
                    let i = rng.integer(k - 1);
                    let above = if i > 0 {
                        log_depths[i - 1] + bounds.gap
                    } else {
                        bounds.u_low
                    };
                    let below = if i < k - 2 {
                        log_depths[i + 1] - bounds.gap
                    } else {
                        bounds.u_high
                    };
                    if below <= above {
                        continue;
                    }
                    log_depths[i] = rng.uniform(above, below);
                }
                Move::Vs => {
                    let j = rng.integer(k);
                    log_vs[j] = reflect(
                        log_vs[j] + copy.steps.vs * rng.normal(),
                        bounds.w_low,
                        bounds.w_high,
                    );
                }
                Move::Noise => {
                    log_noise = reflect(log_noise + copy.steps.noise * rng.normal(), s_low, s_high);
                }
                Move::Shift => {
                    let delta = copy.steps.shift * rng.normal();
                    for w in &mut log_vs {
                        *w += delta;
                    }
                }
                Move::Stretch => {
                    if k <= 1 {
                        continue;
                    }
                    let delta = copy.steps.stretch * rng.normal();
                    for x in &mut log_depths {
                        *x += delta;
                    }
                }
            }
            if t == 0 {
                proposed[index] += 1;
            }
            let mut ok = false;
            if let Some((predicted, value, likelihood)) =
                evaluate(space, &bounds, &log_depths, &log_vs, log_noise)
            {
                ok = (rng.random() + 1e-300).ln()
                    < betas[t] * (likelihood - copies[t].likelihood) + extra;
                if ok {
                    if value < best_misfit {
                        best_misfit = value;
                        best_depths = log_depths.iter().map(|x| x.exp()).collect();
                        best_vs = log_vs.iter().map(|w| w.exp()).collect();
                    }
                    let copy = &mut copies[t];
                    copy.log_depths = log_depths;
                    copy.log_vs = log_vs;
                    copy.log_noise = log_noise;
                    copy.predicted = predicted;
                    copy.misfit = value;
                    copy.likelihood = likelihood;
                }
            }
            if t == 0 && ok {
                accepted[index] += 1;
            }
            if burning && let Some(size) = copies[t].steps.of(step) {
                *size *= (gain * (if ok { 1.0 } else { 0.0 } - TARGET)).exp();
            }
        }

        // Exchanges between neighbouring copies, the even pairs then the odd ones in turn.
        let mut a = iteration % 2;
        while a + 1 < k_count {
            swaps.1 += 1;
            let tried = (rng.random() + 1e-300).ln()
                < (betas[a] - betas[a + 1]) * (copies[a + 1].likelihood - copies[a].likelihood);
            if tried {
                swaps.0 += 1;
                // The models change places; each copy keeps its temperature's steps.
                let (cold, hot) = copies.split_at_mut(a + 1);
                let (cold, hot) = (&mut cold[a], &mut hot[0]);
                std::mem::swap(&mut cold.log_depths, &mut hot.log_depths);
                std::mem::swap(&mut cold.log_vs, &mut hot.log_vs);
                std::mem::swap(&mut cold.log_noise, &mut hot.log_noise);
                std::mem::swap(&mut cold.misfit, &mut hot.misfit);
                std::mem::swap(&mut cold.likelihood, &mut hot.likelihood);
                std::mem::swap(&mut cold.predicted, &mut hot.predicted);
            }
            a += 2;
        }

        if !burning && (iteration - settings.n_burnin) % settings.save_every == 0 {
            let cold = &copies[0];
            kept.push((
                cold.log_depths.iter().map(|x| x.exp()).collect(),
                cold.log_vs.iter().map(|w| w.exp()).collect(),
                cold.log_noise.exp(),
                cold.predicted.clone(),
            ));
        }
    }

    let count = kept.len();
    let width = space.max_layers - 1;
    let mut depths = vec![f64::NAN; count * width];
    let mut vs = vec![f64::NAN; count * space.max_layers];
    let mut predicted: Vec<Vec<f64>> = space
        .curves
        .iter()
        .map(|curve| Vec::with_capacity(count * curve.periods.len()))
        .collect();
    let mut layers = Vec::with_capacity(count);
    let mut noise = Vec::with_capacity(count);
    let mut rms_kept = Vec::with_capacity(count);
    for (row, (d, v, n, p)) in kept.iter().enumerate() {
        depths[row * width..row * width + d.len()].copy_from_slice(d);
        vs[row * space.max_layers..row * space.max_layers + v.len()].copy_from_slice(v);
        layers.push(v.len());
        noise.push(*n);
        rms_kept.push(rms(&space.curves, p));
        for (curve, values) in predicted.iter_mut().zip(p) {
            curve.extend_from_slice(values);
        }
    }
    Ok(Chain {
        layers,
        depths,
        vs,
        noise,
        predicted,
        rms: rms_kept,
        proposed,
        accepted,
        swaps,
        steps: copies[0].steps,
        best_depths,
        best_vs,
        best_misfit,
    })
}
