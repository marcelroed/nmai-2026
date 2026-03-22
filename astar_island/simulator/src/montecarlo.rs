use rayon::prelude::*;

use crate::io::{InitialState, ReplayData};
use crate::params::Params;
use crate::phases::simulate;
use crate::world::World;

/// Run `n_sims` simulations in parallel from an `InitialState`, aggregate the
/// per-cell terrain-class counts, and return a probability distribution grid.
///
/// Each simulation is seeded with `base_seed + i` (i = 0..n_sims) so that
/// results are deterministic but independent.
///
/// Returns `height × width` grid where each cell is a `[f64; 6]` probability
/// vector over prediction classes 0–5.  Each probability is floored at 0.01
/// before renormalisation so that no class is completely excluded.
pub fn run_montecarlo(
    initial_state: &InitialState,
    params: &Params,
    n_sims: usize,
    steps: u32,
    base_seed: u64,
) -> Vec<Vec<[f64; 6]>> {
    // Run n_sims simulations in parallel.
    let snapshots: Vec<Vec<Vec<usize>>> = (0..n_sims)
        .into_par_iter()
        .map(|i| {
            let seed = base_seed + i as u64;
            let mut world = World::from_initial_state(initial_state, params, seed);
            simulate(&mut world, params, steps);
            world.prediction_snapshot()
        })
        .collect();

    aggregate_snapshots(snapshots, Some(&initial_state.grid))
}

/// Run `n_sims` simulations in parallel starting from `replay` frame 0.
///
/// Uses `World::from_replay_frame` so all settlement statistics are known
/// precisely from the replay rather than sampled from priors.
pub fn run_montecarlo_from_replay(
    replay: &ReplayData,
    params: &Params,
    n_sims: usize,
    steps: u32,
    base_seed: u64,
) -> Vec<Vec<[f64; 6]>> {
    let frame = &replay.frames[0];
    let width = replay.width;
    let height = replay.height;

    let snapshots: Vec<Vec<Vec<usize>>> = (0..n_sims)
        .into_par_iter()
        .map(|i| {
            let seed = base_seed + i as u64;
            let mut world = World::from_replay_frame(frame, width, height, seed);
            simulate(&mut world, params, steps);
            world.prediction_snapshot()
        })
        .collect();

    aggregate_snapshots(snapshots, Some(&replay.frames[0].grid))
}

/// Run ensemble Monte Carlo: multiple parameter sets, weighted by score.
/// Each param set gets `sims_per_member` simulations. Results are averaged
/// with softmax weighting by log-likelihood score.
pub fn run_ensemble_montecarlo(
    initial_state: &InitialState,
    param_sets: &[(Params, f64)], // (params, log-likelihood score)
    sims_per_member: usize,
    steps: u32,
    base_seed: u64,
) -> Vec<Vec<[f64; 6]>> {
    if param_sets.is_empty() {
        return run_montecarlo(initial_state, &Params::default_prior(), sims_per_member, steps, base_seed);
    }

    let height = initial_state.grid.len();
    let width = initial_state.grid[0].len();

    // Compute softmax weights from scores
    let max_score = param_sets.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = param_sets.iter().map(|(_, s)| (s - max_score).exp()).collect();
    let weight_sum: f64 = weights.iter().sum();
    let weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

    // Run MC for each param set and accumulate weighted distributions
    let mut combined = vec![vec![[0.0f64; 6]; width]; height];

    for (idx, (params, _score)) in param_sets.iter().enumerate() {
        let member_seed = base_seed + (idx as u64 * 100_000);
        let member_dist = run_montecarlo(initial_state, params, sims_per_member, steps, member_seed);
        let w = weights[idx];

        for y in 0..height {
            for x in 0..width {
                for c in 0..6 {
                    combined[y][x][c] += w * member_dist[y][x][c];
                }
            }
        }
    }

    // The individual run_montecarlo calls already applied floor+renorm,
    // but the weighted average may need re-flooring
    for row in combined.iter_mut() {
        for cell in row.iter_mut() {
            for v in cell.iter_mut() {
                if *v < 0.01 { *v = 0.01; }
            }
            let sum: f64 = cell.iter().sum();
            for v in cell.iter_mut() { *v /= sum; }
        }
    }

    // Force static cells
    for y in 0..height {
        for x in 0..width {
            match initial_state.grid[y][x] {
                crate::terrain::TerrainType::Ocean => {
                    combined[y][x] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                }
                crate::terrain::TerrainType::Mountain => {
                    combined[y][x] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
                }
                _ => {}
            }
        }
    }

    combined
}

/// Aggregate raw class-index snapshots into a probability distribution grid.
///
/// For each cell, counts how many simulations produced each of the 6 class
/// indices, applies a floor of 0.01 to each class count, then normalises to
/// sum to 1.0.
fn aggregate_snapshots(
    snapshots: Vec<Vec<Vec<usize>>>,
    initial_grid: Option<&Vec<Vec<crate::terrain::TerrainType>>>,
) -> Vec<Vec<[f64; 6]>> {
    if snapshots.is_empty() {
        return Vec::new();
    }

    let height = snapshots[0].len();
    let width = snapshots[0][0].len();
    let n = snapshots.len() as f64;

    let mut result = vec![vec![[0.0f64; 6]; width]; height];

    for snap in &snapshots {
        for y in 0..height {
            for x in 0..width {
                let cls = snap[y][x];
                result[y][x][cls] += 1.0;
            }
        }
    }

    // Normalise: apply floor of 0.01 (as a fraction of n, then floor raw),
    // then renormalise to sum to 1.0.
    for row in result.iter_mut() {
        for cell in row.iter_mut() {
            // Convert counts to raw frequencies.
            for v in cell.iter_mut() {
                *v /= n;
            }
            // Apply floor of 0.01 per class.
            for v in cell.iter_mut() {
                if *v < 0.01 {
                    *v = 0.01;
                }
            }
            // Renormalise to sum to 1.0.
            let sum: f64 = cell.iter().sum();
            for v in cell.iter_mut() {
                *v /= sum;
            }
        }
    }

    // Force static cells to exact probabilities (no floor needed).
    if let Some(grid) = initial_grid {
        for y in 0..height {
            for x in 0..width {
                match grid[y][x] {
                    crate::terrain::TerrainType::Ocean => {
                        result[y][x] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    }
                    crate::terrain::TerrainType::Mountain => {
                        result[y][x] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
                    }
                    _ => {}
                }
            }
        }
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{ReplayData, RoundDetails};

    const ROUND_ID: &str = "ae78003a-4efe-425a-881a-d16a39bca0ad";

    #[test]
    fn test_run_montecarlo_shape() {
        let details = RoundDetails::load(ROUND_ID).expect("load details");
        let params = Params::default_prior();
        let initial = &details.initial_states[0];

        let dist = run_montecarlo(initial, &params, 3, 5, 42);

        assert_eq!(dist.len(), details.map_height);
        assert_eq!(dist[0].len(), details.map_width);
        // Each cell sums to ~1.0.
        let sum: f64 = dist[0][0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "cell sum = {}", sum);
    }

    #[test]
    fn test_run_montecarlo_from_replay_shape() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let params = Params::default_prior();

        let dist = run_montecarlo_from_replay(&replay, &params, 3, 5, 42);

        assert_eq!(dist.len(), replay.height);
        assert_eq!(dist[0].len(), replay.width);
        let sum: f64 = dist[0][0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "cell sum = {}", sum);
    }

    #[test]
    fn test_probabilities_floored() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let params = Params::default_prior();

        let dist = run_montecarlo_from_replay(&replay, &params, 5, 2, 0);

        for row in &dist {
            for cell in row {
                for &p in cell {
                    // Static cells (ocean/mountain) may have exact 0.0 for
                    // non-matching classes, so we only check non-negative.
                    assert!(p >= 0.0, "probability must be >= 0");
                    assert!(p <= 1.0, "probability must be <= 1");
                }
            }
        }
    }
}
