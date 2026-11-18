//! The samplers' random numbers: PCG64 (rand_pcg's Lcg128Xsl64), seeded from a chain's seed, so
//! that a seed gives the same chain from run to run.

use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;

pub struct Random(Pcg64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(Pcg64::seed_from_u64(seed))
    }

    /// Uniform in [0, 1).
    pub fn random(&mut self) -> f64 {
        self.0.random::<f64>()
    }

    /// Uniform in [low, high).
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }

    pub fn normal(&mut self) -> f64 {
        self.0.sample(StandardNormal)
    }

    /// Uniform in 0..n.
    pub fn integer(&mut self, n: usize) -> usize {
        self.0.random_range(0..n)
    }

    /// An index drawn with the probabilities `weights` (summing to one), as numpy's choice with
    /// p: the first whose cumulative share exceeds a uniform draw.
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let u = self.random();
        let mut cumulative = 0.0;
        for (i, weight) in weights.iter().enumerate() {
            cumulative += weight / total;
            if u < cumulative {
                return i;
            }
        }
        weights.len() - 1
    }

    /// `k` distinct indices of 0..n, in random order (Floyd's sampling, then shuffled).
    pub fn distinct(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut chosen: Vec<usize> = Vec::with_capacity(k);
        for j in n - k..n {
            let t = self.integer(j + 1);
            chosen.push(if chosen.contains(&t) { j } else { t });
        }
        for i in (1..chosen.len()).rev() {
            let j = self.integer(i + 1);
            chosen.swap(i, j);
        }
        chosen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_gives_the_same_numbers() {
        let (mut a, mut b) = (Random::new(7), Random::new(7));
        for _ in 0..100 {
            assert_eq!(a.random().to_bits(), b.random().to_bits());
            assert_eq!(a.normal().to_bits(), b.normal().to_bits());
        }
    }

    #[test]
    fn distinct_indices_are_distinct_and_spread() {
        let mut rng = Random::new(3);
        let mut seen = [0_usize; 10];
        for _ in 0..20_000 {
            let picks = rng.distinct(10, 4);
            let mut sorted = picks.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 4);
            seen[picks[0]] += 1;
        }
        // The first pick uniform over the ten: 2000 each, give or take.
        assert!(
            seen.iter().all(|&count| (1_700..2_300).contains(&count)),
            "{seen:?}"
        );
    }

    #[test]
    fn weighted_draws_follow_their_weights() {
        let mut rng = Random::new(5);
        let mut counts = [0_usize; 3];
        for _ in 0..30_000 {
            counts[rng.weighted(&[0.2, 0.3, 0.5])] += 1;
        }
        assert!((5_600..6_400).contains(&counts[0]), "{counts:?}");
        assert!((14_400..15_600).contains(&counts[2]), "{counts:?}");
    }
}
