//! Layered Vs models with the layers given: the chains run together by DREAM(ZS). sigpipe's
//! dream.py documents the method (ter Braak and Vrugt 2008, Vrugt 2016).

use crate::data::{Curve, allowed, log_likelihood, misfit, n_points, phase_velocities, rms};
use crate::rng::Random;

/// DREAM(ZS)'s settings (Vrugt 2016): pairs of past states summed in a proposal, the jump
/// between modes, the snooker's share, the archive's pace, the crossover values, and the noise
/// added.
const PAIRS: usize = 3;
const JUMP_EVERY: usize = 5;
const SNOOKER: f64 = 0.1;
const ARCHIVE_EVERY: usize = 10;
const CROSSOVERS: usize = 3;
const SPREAD: f64 = 0.1; // each difference scaled by 1 +- this
const JITTER: f64 = 1e-6; // added to each move, on the unit cube

/// The curves, the parameters sampled and their priors, and where each value goes.
pub struct Problem {
    pub curves: Vec<Curve>,
    pub low: Vec<f64>, // each sampled parameter's prior bounds
    pub high: Vec<f64>,
    pub vs_template: Vec<f64>, // the layers' Vs, the fixed ones in place
    pub thickness_template: Vec<f64>,
    pub vs_from: Vec<usize>, // the sampled values' places in them
    pub vs_to: Vec<usize>,
    pub thickness_from: Vec<usize>,
    pub thickness_to: Vec<usize>,
    pub vs_index: Vec<usize>, // the sampled parameters that are a Vs
    pub vp_vs: f64,
    pub least_ratio: f64,
    pub noise_bounds: (f64, f64), // the noise factor's prior (data.py's NOISE_BOUNDS)
}

impl Problem {
    fn layers(&self, values: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut vs = self.vs_template.clone();
        for (&from, &to) in self.vs_from.iter().zip(&self.vs_to) {
            vs[to] = values[from];
        }
        let mut thickness = self.thickness_template.clone();
        for (&from, &to) in self.thickness_from.iter().zip(&self.thickness_to) {
            thickness[to] = values[from];
        }
        (thickness, vs)
    }

    fn predict(&self, values: &[f64]) -> Option<Vec<Vec<f64>>> {
        let (thickness, vs) = self.layers(values);
        if !allowed(&vs, self.least_ratio) {
            return None;
        }
        phase_velocities(&self.curves, &thickness, &vs, self.vp_vs)
    }
}

pub struct Settings {
    pub n_iterations: usize,
    pub n_burnin: usize,
    pub save_every: usize,
    pub n_chains: usize,
}

/// One chain's kept states (on the unit cube, the noise factor last), their curves and fit, and
/// how it moved.
pub struct ChainOut {
    pub units: Vec<f64>,          // kept x dims
    pub predicted: Vec<Vec<f64>>, // per curve: kept x points, by increasing period
    pub rms: Vec<f64>,
    pub accepted: usize,
    pub squared_moves: Vec<f64>, // dims
}

pub struct Sampled {
    pub chains: Vec<ChainOut>,
    pub best_unit: Vec<f64>,
    pub best_predicted: Vec<Vec<f64>>,
    pub crossover: [f64; CROSSOVERS], // each crossover value's final probability
}

/// A model of the priors on the unit cube, its Vs increasing where their bounds allow.
fn prior_draw(problem: &Problem, rng: &mut Random) -> Vec<f64> {
    let mut values: Vec<f64> = problem
        .low
        .iter()
        .zip(&problem.high)
        .map(|(&low, &high)| rng.uniform(low, high))
        .collect();
    if problem.least_ratio > 0.0 && !problem.vs_index.is_empty() {
        let mut sorted: Vec<f64> = problem.vs_index.iter().map(|&i| values[i]).collect();
        sorted.sort_by(f64::total_cmp);
        for (&i, value) in problem.vs_index.iter().zip(sorted) {
            values[i] = value.clamp(problem.low[i], problem.high[i]);
        }
    }
    values
        .iter()
        .zip(problem.low.iter().zip(&problem.high))
        .map(|(value, (low, high))| (value / low).ln() / (high / low).ln())
        .collect()
}

/// A model of the priors (unit cube) whose modes exist over the curves' periods, and its
/// predicted curves.
fn start(problem: &Problem, rng: &mut Random) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    for _ in 0..10_000 {
        let u = prior_draw(problem, rng);
        let values: Vec<f64> = u
            .iter()
            .zip(problem.low.iter().zip(&problem.high))
            .map(|(u, (low, high))| low * (high / low).powf(*u))
            .collect();
        if let Some(predicted) = problem.predict(&values) {
            return Ok((u, predicted));
        }
    }
    Err(
        "No model of the priors has the picked modes over their frequencies, within the Vs ratios \
         allowed: check that each layer's bounds hold the curves."
            .to_string(),
    )
}

/// A move along the line through `x` and an archived state, by the projected difference of two
/// others, and its log acceptance factor (ter Braak and Vrugt 2008); -inf where none is made.
fn snooker(x: &[f64], archive: &[f64], dims: usize, rng: &mut Random) -> (Vec<f64>, f64) {
    let picks = rng.distinct(archive.len() / dims, 3);
    let row = |i: usize| &archive[picks[i] * dims..(picks[i] + 1) * dims];
    let (z, z1, z2) = (row(0), row(1), row(2));
    let direction: Vec<f64> = x.iter().zip(z).map(|(a, b)| a - b).collect();
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return (x.to_vec(), f64::NEG_INFINITY);
    }
    let unit: Vec<f64> = direction.iter().map(|v| v / norm).collect();
    let gamma = rng.uniform(1.2, 2.2);
    let projection: f64 = z1
        .iter()
        .zip(z2)
        .zip(&unit)
        .map(|((a, b), u)| (a - b) * u)
        .sum();
    let candidate: Vec<f64> = x
        .iter()
        .zip(&unit)
        .map(|(v, u)| v + gamma * projection * u)
        .collect();
    if candidate.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        // Outside the priors: no reflection, which would bend the line.
        return (x.to_vec(), f64::NEG_INFINITY);
    }
    let after = candidate
        .iter()
        .zip(z)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    if after < 1e-12 {
        return (x.to_vec(), f64::NEG_INFINITY);
    }
    (candidate, (dims as f64 - 1.0) * (after / norm).ln())
}

/// `u` folded back into the unit cube, as by mirrors at its faces.
fn reflect(u: &mut [f64]) {
    for value in u.iter_mut() {
        let folded = value.rem_euclid(2.0);
        *value = if folded > 1.0 { 2.0 - folded } else { folded };
    }
}

/// The chains together, from models of the priors (DREAM(ZS)).
pub fn sample(problem: &Problem, settings: &Settings, seed: u64) -> Result<Sampled, String> {
    let mut rng = Random::new(seed);
    let names = problem.low.len();
    let dims = names + 1; // the noise factor last, as its logarithm
    let mut low: Vec<f64> = problem.low.iter().map(|v| v.ln()).collect();
    low.push(problem.noise_bounds.0.ln());
    let mut high: Vec<f64> = problem.high.iter().map(|v| v.ln()).collect();
    high.push(problem.noise_bounds.1.ln());
    let span: Vec<f64> = high.iter().zip(&low).map(|(h, l)| h - l).collect();
    let n_points = n_points(&problem.curves);
    let n_chains = settings.n_chains;

    // The values on the unit cube's state, the noise factor's logarithm last.
    let from_unit = |u: &[f64]| -> Vec<f64> {
        (0..dims)
            .map(|i| {
                let x = low[i] + u[i] * span[i];
                if i < names { x.exp() } else { x }
            })
            .collect()
    };
    // The likelihood's logarithm and the priors', uniform in the values (their logarithms'
    // Jacobian) and in the noise factor's logarithm.
    let log_posterior = |misfit_value: f64, u: &[f64]| -> f64 {
        let jacobian: f64 = (0..names).map(|i| low[i] + u[i] * span[i]).sum();
        let log_noise = low[names] + u[names] * span[names];
        log_likelihood(misfit_value, log_noise, n_points) + jacobian
    };

    // The archive: draws of the priors to begin with, then the chains' states.
    let mut size = (10 * dims).max(100);
    let mut archive: Vec<f64> =
        Vec::with_capacity((size + n_chains * (settings.n_iterations / ARCHIVE_EVERY + 1)) * dims);
    for _ in 0..size {
        archive.extend(prior_draw(problem, &mut rng));
        archive.push(rng.random());
    }

    let mut units: Vec<Vec<f64>> = Vec::with_capacity(n_chains);
    let mut misfits: Vec<f64> = Vec::with_capacity(n_chains);
    let mut predictions: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_chains);
    let mut lls: Vec<f64> = Vec::with_capacity(n_chains);
    for _ in 0..n_chains {
        let (mut u, predicted) = start(problem, &mut rng)?;
        u.push(1.0); // the noise factor starts at the picks' own
        let value = misfit(&problem.curves, &predicted);
        lls.push(log_posterior(value, &u));
        units.push(u);
        misfits.push(value);
        predictions.push(predicted);
    }

    let mut best = 0;
    for (c, value) in misfits.iter().enumerate() {
        if *value < misfits[best] {
            best = c;
        }
    }
    let mut best_misfit = misfits[best];
    let mut best_unit = units[best].clone();
    let mut best_predicted = predictions[best].clone();
    let mut accepted = vec![0_usize; n_chains];
    let mut squared_moves = vec![vec![0.0; dims]; n_chains];
    let mut kept_units: Vec<Vec<f64>> = vec![Vec::new(); n_chains];
    let mut kept_predicted: Vec<Vec<Vec<f64>>> =
        vec![vec![Vec::new(); problem.curves.len()]; n_chains];
    let mut kept_rms: Vec<Vec<f64>> = vec![Vec::new(); n_chains];
    // The crossover values, their probabilities, and each one's squared jumps and uses, which
    // adapt the probabilities during the burn-in.
    let crossover: [f64; CROSSOVERS] = std::array::from_fn(|i| (i + 1) as f64 / CROSSOVERS as f64);
    let mut p_crossover = [1.0 / CROSSOVERS as f64; CROSSOVERS];
    let mut jumped = [0.0; CROSSOVERS];
    let mut used = [0.0; CROSSOVERS];

    let mut first = 0; // the archive's first state proposals draw from
    let mut middle: Option<usize> = None;
    for iteration in 1..=settings.n_iterations {
        let burning = iteration <= settings.n_burnin;
        if iteration == settings.n_burnin / 2 + 1 {
            middle = Some(size);
        }
        if iteration == settings.n_burnin + 1
            && let Some(middle) = middle
            && size - middle > 4 * PAIRS
        {
            first = middle;
        }
        // The chains' spread, per parameter.
        let spread: Vec<f64> = (0..dims)
            .map(|i| {
                let mean = units.iter().map(|u| u[i]).sum::<f64>() / n_chains as f64;
                let variance = units
                    .iter()
                    .map(|u| (u[i] - mean) * (u[i] - mean))
                    .sum::<f64>()
                    / n_chains as f64;
                variance.sqrt() + 1e-12
            })
            .collect();
        for c in 0..n_chains {
            let mut crossing: Option<usize> = None;
            let (candidate, factor) = if rng.random() < SNOOKER {
                snooker(
                    &units[c],
                    &archive[first * dims..size * dims],
                    dims,
                    &mut rng,
                )
            } else {
                let m = rng.weighted(&p_crossover);
                crossing = Some(m);
                let mut mask: Vec<bool> = (0..dims).map(|_| rng.random() < crossover[m]).collect();
                if !mask.iter().any(|&moving| moving) {
                    let i = rng.integer(dims);
                    mask[i] = true;
                }
                let moved = mask.iter().filter(|&&moving| moving).count();
                let pairs = 1 + rng.integer(PAIRS);
                let jump = iteration % JUMP_EVERY == 0;
                let gamma = if jump {
                    1.0
                } else {
                    2.38 / ((2 * pairs * moved) as f64).sqrt()
                };
                let picks: Vec<usize> = rng
                    .distinct(size - first, 2 * pairs)
                    .into_iter()
                    .map(|p| first + p)
                    .collect();
                let difference: Vec<f64> = (0..dims)
                    .map(|i| {
                        let plus: f64 = picks[..pairs].iter().map(|&p| archive[p * dims + i]).sum();
                        let minus: f64 =
                            picks[pairs..].iter().map(|&p| archive[p * dims + i]).sum();
                        plus - minus
                    })
                    .collect();
                let scale: Vec<f64> = (0..moved)
                    .map(|_| 1.0 + rng.uniform(-SPREAD, SPREAD))
                    .collect();
                let noise: Vec<f64> = (0..moved).map(|_| rng.normal()).collect();
                let mut candidate = units[c].clone();
                let mut j = 0;
                for i in 0..dims {
                    if mask[i] {
                        candidate[i] += gamma * scale[j] * difference[i] + JITTER * noise[j];
                        j += 1;
                    }
                }
                reflect(&mut candidate);
                (candidate, 0.0)
            };

            // A snooker that made no move is rejected whatever its fit: its curves are spared.
            let mut accept = false;
            let mut moved_to: Option<(Vec<Vec<f64>>, f64, f64)> = None;
            if factor > f64::NEG_INFINITY {
                let z = from_unit(&candidate);
                if let Some(predicted) = problem.predict(&z[..names]) {
                    let value = misfit(&problem.curves, &predicted);
                    let ll = log_posterior(value, &candidate);
                    accept = (rng.random() + 1e-300).ln() < ll - lls[c] + factor;
                    moved_to = Some((predicted, value, ll));
                }
            }
            if let Some(m) = crossing
                && burning
            {
                used[m] += 1.0;
                if accept {
                    jumped[m] += candidate
                        .iter()
                        .zip(&units[c])
                        .zip(&spread)
                        .map(|((a, b), s)| ((a - b) / s) * ((a - b) / s))
                        .sum::<f64>();
                }
            }
            if accept && let Some((predicted, value, ll)) = moved_to {
                for ((total, a), b) in squared_moves[c].iter_mut().zip(&candidate).zip(&units[c]) {
                    *total += (a - b) * (a - b);
                }
                accepted[c] += 1;
                if value < best_misfit {
                    best_misfit = value;
                    best_unit = candidate.clone();
                    best_predicted = predicted.clone();
                }
                units[c] = candidate;
                misfits[c] = value;
                predictions[c] = predicted;
                lls[c] = ll;
            }
            if !burning && (iteration - settings.n_burnin) % settings.save_every == 0 {
                kept_units[c].extend_from_slice(&units[c]);
                for (curve, values) in kept_predicted[c].iter_mut().zip(&predictions[c]) {
                    curve.extend_from_slice(values);
                }
                kept_rms[c].push(rms(&problem.curves, &predictions[c]));
            }
        }
        if burning && iteration % 10 == 0 && used.iter().all(|&count| count > 0.0) {
            // The crossover values that move the chains furthest per use, more often.
            let rates: [f64; CROSSOVERS] = std::array::from_fn(|i| jumped[i] / used[i]);
            let total: f64 = rates.iter().sum();
            if total > 0.0 {
                for (p, rate) in p_crossover.iter_mut().zip(rates) {
                    *p = (rate / total).max(0.05);
                }
                let sum: f64 = p_crossover.iter().sum();
                for p in &mut p_crossover {
                    *p /= sum;
                }
            }
        }
        if iteration % ARCHIVE_EVERY == 0 {
            for u in &units {
                archive.extend_from_slice(u);
                size += 1;
            }
        }
    }

    let chains = (0..n_chains)
        .map(|c| ChainOut {
            units: std::mem::take(&mut kept_units[c]),
            predicted: std::mem::take(&mut kept_predicted[c]),
            rms: std::mem::take(&mut kept_rms[c]),
            accepted: accepted[c],
            squared_moves: squared_moves[c].clone(),
        })
        .collect();
    Ok(Sampled {
        chains,
        best_unit,
        best_predicted,
        crossover: p_crossover,
    })
}
