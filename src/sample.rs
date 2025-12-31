//! Token sampling with temperature and top-p.

use crate::ops::softmax;
use rand::Rng;

/// Used for sorting probabilities in top-p sampling.
#[derive(Clone, Copy)]
pub struct ProbIndex {
    pub prob: f32,
    pub index: usize,
}

/// Sample a token from logits with temperature and top-p.
///
/// - `temp == 0`: greedy (argmax)
/// - `topp <= 0 || topp >= 1`: standard multinomial sampling
/// - otherwise: nucleus (top-p) sampling
pub fn sample<R: Rng>(logits: &mut [f32], temp: f64, topp: f64, rng: &mut R) -> i32 {
    // Greedy decoding
    if temp == 0.0 {
        return argmax(logits) as i32;
    }

    // Scale by temperature
    let temp_f32 = temp as f32;
    for l in logits.iter_mut() {
        *l /= temp_f32;
    }
    softmax(logits);

    let r: f32 = rng.random();

    // Standard multinomial sampling
    if topp <= 0.0 || topp >= 1.0 {
        let mut cdf = 0.0f32;
        for (i, &p) in logits.iter().enumerate() {
            cdf += p;
            if r < cdf {
                return i as i32;
            }
        }
        return (logits.len() - 1) as i32;
    }

    // Top-p (nucleus) sampling
    let mut prob_index: Vec<ProbIndex> = logits
        .iter()
        .enumerate()
        .map(|(i, &p)| ProbIndex { prob: p, index: i })
        .collect();

    // Sort descending by probability
    prob_index.sort_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find cutoff
    let topp_f32 = topp as f32;
    let mut cum_prob = 0.0f32;
    let mut last_idx = prob_index.len() - 1;
    for (i, pi) in prob_index.iter().enumerate() {
        cum_prob += pi.prob;
        if cum_prob > topp_f32 {
            last_idx = i;
            break;
        }
    }

    // Sample from truncated distribution
    let r_scaled = r * cum_prob;
    let mut cdf = 0.0f32;
    for pi in prob_index.iter().take(last_idx + 1) {
        cdf += pi.prob;
        if r_scaled < cdf {
            return pi.index as i32;
        }
    }

    prob_index[last_idx].index as i32
}

/// Returns the index of the maximum element.
#[inline]
fn argmax(x: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}
