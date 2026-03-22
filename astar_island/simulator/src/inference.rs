use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::io::InitialState;
use crate::params::Params;
use crate::phases;
use crate::terrain::TerrainType;
use crate::world::World;

// ── Observation ───────────────────────────────────────────────────────────────

/// A single viewport observation from a query.
pub struct Observation {
    pub seed_index: usize,
    pub vx: usize,
    pub vy: usize,
    pub vw: usize,
    pub vh: usize,
    pub grid: Vec<Vec<TerrainType>>,
}

// ── InferenceConfig ───────────────────────────────────────────────────────────

pub struct InferenceConfig {
    pub n_generations: usize,
    pub population_size: usize,
    pub sims_per_candidate: usize,
    pub steps: u32,
    pub early_stop_patience: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            n_generations: 100,
            population_size: 20,
            sims_per_candidate: 25,
            steps: 50,
            early_stop_patience: 20,
        }
    }
}

// ── Pre-processed observation data ───────────────────────────────────────────

/// Flattened cell observation: (map_y, map_x, prediction_class).
struct ObsCell {
    map_y: usize,
    map_x: usize,
    class: usize,
}

/// All observations for a single seed, pre-processed for fast evaluation.
struct SeedObs {
    seed_index: usize,
    cells: Vec<ObsCell>,
}

/// Pre-process observations into per-seed flattened cell lists.
fn preprocess_observations(observations: &[Observation]) -> Vec<SeedObs> {
    let max_seed = match observations.iter().map(|o| o.seed_index).max() {
        Some(m) => m,
        None => return Vec::new(),
    };

    let mut result = Vec::new();
    for seed_idx in 0..=max_seed {
        let mut cells = Vec::new();
        for obs in observations.iter().filter(|o| o.seed_index == seed_idx) {
            for row in 0..obs.vh {
                for col in 0..obs.vw {
                    cells.push(ObsCell {
                        map_y: obs.vy + row,
                        map_x: obs.vx + col,
                        class: obs.grid[row][col].prediction_class(),
                    });
                }
            }
        }
        if !cells.is_empty() {
            result.push(SeedObs { seed_index: seed_idx, cells });
        }
    }
    result
}

// ── Gaussian noise ───────────────────────────────────────────────────────────

#[inline]
fn box_muller(u1: f32, u2: f32) -> f32 {
    let u1 = u1.max(1e-10);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ── Candidate evaluation ─────────────────────────────────────────────────────

/// Evaluate a candidate by simulating and comparing to observations.
/// Returns average log-likelihood per observed cell (higher = better).
fn evaluate_params(
    params: &Params,
    initial_states: &[InitialState],
    seed_obs: &[SeedObs],
    config: &InferenceConfig,
) -> f64 {
    let n_sims = config.sims_per_candidate;
    let mut total_ll = 0.0_f64;
    let mut total_cells = 0_usize;

    for so in seed_obs {
        let initial = match initial_states.get(so.seed_index) {
            Some(s) => s,
            None => continue,
        };

        // Run sims in parallel and only track the cells we need.
        // fold+reduce avoids per-sim allocations: each rayon chunk reuses one counts vec.
        let n_cells = so.cells.len();
        let counts: Vec<u32> = (0..n_sims)
            .into_par_iter()
            .fold(
                || vec![0u32; n_cells],
                |mut counts, i| {
                    let seed = (so.seed_index as u64) * 100_000 + i as u64;
                    let mut world = World::from_initial_state(initial, params, seed);
                    phases::simulate(&mut world, params, config.steps);

                    for (ci, cell) in so.cells.iter().enumerate() {
                        if let Some(row) = world.grid.get(cell.map_y) {
                            if let Some(terrain) = row.get(cell.map_x) {
                                if terrain.prediction_class() == cell.class {
                                    counts[ci] += 1;
                                }
                            }
                        }
                    }
                    counts
                },
            )
            .reduce(
                || vec![0u32; n_cells],
                |mut a, b| {
                    for i in 0..a.len() {
                        a[i] += b[i];
                    }
                    a
                },
            );

        // Compute log-likelihood
        let n_f = n_sims as f64;
        for &c in &counts {
            let freq = (c as f64 / n_f).max(0.01);
            total_ll += freq.ln();
            total_cells += 1;
        }
    }

    if total_cells == 0 { return 0.0; }
    total_ll / total_cells as f64
}

// ── infer_params ─────────────────────────────────────────────────────────────

/// Run evolution strategy to infer hidden parameters from observations.
pub fn infer_params(
    initial_states: &[InitialState],
    observations: &[Observation],
    config: &InferenceConfig,
) -> Params {
    use std::time::Instant;

    let lower = Params::lower_bounds();
    let upper = Params::upper_bounds();
    let n_dims = lower.len();

    // Pre-process observations once
    let seed_obs = preprocess_observations(observations);
    if seed_obs.is_empty() {
        return Params::default_prior();
    }

    // Print problem summary.
    let n_seeds_with_obs = seed_obs.len();
    let total_obs_cells: usize = seed_obs.iter().map(|s| s.cells.len()).sum();
    let total_sims = config.n_generations * config.population_size * config.sims_per_candidate;
    eprintln!("── inference ──────────────────────────────────────────────");
    eprintln!(
        "  observations: {}  seeds: {}  observed cells: {}",
        observations.len(), n_seeds_with_obs, total_obs_cells,
    );
    eprintln!(
        "  generations: {}  population: {}  sims/candidate: {}  steps: {}",
        config.n_generations, config.population_size, config.sims_per_candidate, config.steps,
    );
    eprintln!(
        "  total simulations: {}  dims: {}",
        total_sims, n_dims,
    );
    eprintln!("───────────────────────────────────────────────────────────");

    // Initialise mean and sigma.
    let mut mean: Vec<f32> = Params::default_prior().to_vec();
    let mut sigma: Vec<f32> = (0..n_dims)
        .map(|i| 0.10 * (upper[i] - lower[i]).abs())
        .collect();

    let mut best_params = mean.clone();
    let t_start = Instant::now();
    eprintln!("  evaluating initial prior...");
    let mut best_score = evaluate_params(
        &Params::from_vec(&best_params),
        initial_states,
        &seed_obs,
        config,
    );
    eprintln!("  initial prior ll={:.6}  ({:.1}s)", best_score, t_start.elapsed().as_secs_f64());

    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF_CAFE_BABE);

    let mut last_improvement_gen: usize = 0;
    let mut n_restarts = 0u32;
    const MAX_RESTARTS: u32 = 2; // Up to 2 warm restarts after initial run

    for generation in 0..config.n_generations {
        let t_gen = Instant::now();
        // Generate candidates
        let candidates: Vec<Vec<f32>> = (0..config.population_size)
            .map(|_| {
                (0..n_dims)
                    .map(|i| {
                        let u1 = rng.random::<f32>().max(1e-6);
                        let u2 = rng.random::<f32>().max(1e-6);
                        let z = box_muller(u1, u2);
                        (mean[i] + z * sigma[i]).clamp(lower[i], upper[i])
                    })
                    .collect()
            })
            .collect();

        // Evaluate in parallel
        let scores: Vec<f64> = candidates
            .par_iter()
            .map(|c| {
                let p = Params::from_vec(c);
                evaluate_params(&p, initial_states, &seed_obs, config)
            })
            .collect();

        // Sort descending
        let mut indexed: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update best if improved.
        if let Some(&(best_idx, best_gen_score)) = indexed.first() {
            let improved = best_gen_score > best_score;
            if improved {
                best_score = best_gen_score;
                best_params = candidates[best_idx].clone();
                last_improvement_gen = generation;
            }

            let elapsed = t_start.elapsed().as_secs_f64();
            let gen_elapsed = t_gen.elapsed().as_secs_f64();
            let gens_done = generation + 1;
            let gens_left = config.n_generations - gens_done;
            let avg_per_gen = elapsed / gens_done as f64;
            let eta = avg_per_gen * gens_left as f64;
            let stale = generation - last_improvement_gen;

            let avg_sigma: f32 = sigma.iter().sum::<f32>() / sigma.len() as f32;
            let marker = if improved { " *" } else { "" };

            eprintln!(
                "gen {:>3}/{}  best={:.6}  gen={:.6}  σ={:.4}  \
                 [{:.1}s, {:.1}s/gen, eta {:.0}s]{}{}",
                gens_done,
                config.n_generations,
                best_score,
                best_gen_score,
                avg_sigma,
                gen_elapsed,
                avg_per_gen,
                eta,
                marker,
                if stale >= 20 && stale % 10 == 0 {
                    format!("  (stale {})", stale)
                } else {
                    String::new()
                },
            );
        }

        // Early stopping with warm restart
        let stale_generations = generation.saturating_sub(last_improvement_gen);
        if stale_generations >= config.early_stop_patience {
            if n_restarts < MAX_RESTARTS {
                n_restarts += 1;
                eprintln!(
                    "Warm restart #{} at generation {}: resetting sigma, continuing from best",
                    n_restarts, generation
                );
                // Reset to best params found so far, with moderate sigma boost for exploration
                mean = best_params.clone();
                sigma = (0..n_dims)
                    .map(|i| 0.08 * (upper[i] - lower[i]).abs())
                    .collect();
                last_improvement_gen = generation;
                continue;
            } else {
                eprintln!("Early stop at generation {generation}: no improvement for {stale_generations} generations (after {n_restarts} restarts)");
                break;
            }
        }

        // Elite selection (top 25%)
        let n_elites = ((config.population_size as f64 * 0.25).ceil() as usize).max(1);

        // Update mean
        let mut new_mean = vec![0.0_f32; n_dims];
        for &(ei, _) in &indexed[..n_elites] {
            for d in 0..n_dims {
                new_mean[d] += candidates[ei][d];
            }
        }
        let n_elites_f = n_elites as f32;
        for d in 0..n_dims {
            new_mean[d] /= n_elites_f;
        }

        // Update sigma
        let mut new_sigma = vec![0.0_f32; n_dims];
        for &(ei, _) in &indexed[..n_elites] {
            for d in 0..n_dims {
                let diff = candidates[ei][d] - new_mean[d];
                new_sigma[d] += diff * diff;
            }
        }
        for d in 0..n_dims {
            new_sigma[d] = (new_sigma[d] / n_elites_f).sqrt().max(0.001);
        }

        mean = new_mean;
        sigma = new_sigma;
    }

    let total_elapsed = t_start.elapsed().as_secs_f64();
    eprintln!("───────────────────────────────────────────────────────────");
    eprintln!(
        "  done in {:.1}s  best_ll={:.6}  last improvement at gen {}",
        total_elapsed, best_score, last_improvement_gen + 1,
    );
    eprintln!("───────────────────────────────────────────────────────────");

    Params::from_vec(&best_params)
}

/// Scored parameter set for ensemble use.
pub struct ScoredParams {
    pub params: Params,
    pub score: f64,
}

/// Run inference and return the top-K parameter sets with their scores.
/// Used for ensemble Monte Carlo — running MC with multiple param sets
/// and averaging the distributions.
pub fn infer_top_k(
    initial_states: &[InitialState],
    observations: &[Observation],
    config: &InferenceConfig,
    top_k: usize,
) -> Vec<ScoredParams> {
    let lower = Params::lower_bounds();
    let upper = Params::upper_bounds();
    let n_dims = lower.len();

    let seed_obs = preprocess_observations(observations);
    if seed_obs.is_empty() {
        return vec![ScoredParams { params: Params::default_prior(), score: 0.0 }];
    }

    let n_seeds_with_obs = seed_obs.len();
    let total_obs_cells: usize = seed_obs.iter().map(|s| s.cells.len()).sum();
    eprintln!("Ensemble inference: {} seeds, {} cells, top-{top_k}",
              n_seeds_with_obs, total_obs_cells);

    let mut mean: Vec<f32> = Params::default_prior().to_vec();
    let mut sigma: Vec<f32> = (0..n_dims)
        .map(|i| 0.10 * (upper[i] - lower[i]).abs())
        .collect();

    // Track top-K across all generations
    let mut top_candidates: Vec<(Vec<f32>, f64)> = Vec::new();

    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF_CAFE_BABE);

    let mut stale_generations = 0_usize;
    let mut best_score = f64::NEG_INFINITY;
    let start = std::time::Instant::now();

    for generation in 0..config.n_generations {
        let candidates: Vec<Vec<f32>> = (0..config.population_size)
            .map(|_| {
                (0..n_dims)
                    .map(|i| {
                        let u1 = rng.random::<f32>().max(1e-6);
                        let u2 = rng.random::<f32>().max(1e-6);
                        let z = box_muller(u1, u2);
                        (mean[i] + z * sigma[i]).clamp(lower[i], upper[i])
                    })
                    .collect()
            })
            .collect();

        let scores: Vec<f64> = candidates
            .par_iter()
            .map(|c| {
                let p = Params::from_vec(c);
                evaluate_params(&p, initial_states, &seed_obs, config)
            })
            .collect();

        // Add all candidates to the pool
        for (i, score) in scores.iter().enumerate() {
            top_candidates.push((candidates[i].clone(), *score));
        }

        // Keep only top-K
        top_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_candidates.truncate(top_k * 2); // Keep 2x buffer for diversity

        let mut indexed: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let generation_best = indexed[0].1;
        if generation_best > best_score {
            best_score = generation_best;
            stale_generations = 0;
        } else {
            stale_generations += 1;
        }

        if generation % 10 == 0 {
            let elapsed = start.elapsed().as_secs();
            eprintln!("Gen {generation:3}/{}: best_ll={best_score:.4}  [{elapsed}s]",
                      config.n_generations);
        }

        if stale_generations >= config.early_stop_patience {
            eprintln!("Early stop at generation {generation}");
            break;
        }

        let n_elites = ((config.population_size as f64 * 0.25).ceil() as usize).max(1);
        let mut new_mean = vec![0.0_f32; n_dims];
        for &(ei, _) in &indexed[..n_elites] {
            for d in 0..n_dims { new_mean[d] += candidates[ei][d]; }
        }
        let n_f = n_elites as f32;
        for d in 0..n_dims { new_mean[d] /= n_f; }

        let mut new_sigma = vec![0.0_f32; n_dims];
        for &(ei, _) in &indexed[..n_elites] {
            for d in 0..n_dims {
                let diff = candidates[ei][d] - new_mean[d];
                new_sigma[d] += diff * diff;
            }
        }
        for d in 0..n_dims { new_sigma[d] = (new_sigma[d] / n_f).sqrt().max(0.001); }

        mean = new_mean;
        sigma = new_sigma;
    }

    // Return top-K unique candidates
    top_candidates.truncate(top_k);
    let elapsed = start.elapsed().as_secs();
    eprintln!("Ensemble inference complete in {elapsed}s. Top-{top_k} scores: {:?}",
              top_candidates.iter().map(|(_, s)| format!("{s:.4}")).collect::<Vec<_>>());

    top_candidates
        .into_iter()
        .map(|(v, score)| ScoredParams { params: Params::from_vec(&v), score })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_defaults() {
        let cfg = InferenceConfig::default();
        assert_eq!(cfg.n_generations, 100);
        assert_eq!(cfg.population_size, 20);
        assert_eq!(cfg.sims_per_candidate, 25);
        assert_eq!(cfg.steps, 50);
        assert_eq!(cfg.early_stop_patience, 20);
    }

    #[test]
    fn test_infer_params_no_observations_returns_prior() {
        let config = InferenceConfig {
            n_generations: 1,
            population_size: 2,
            sims_per_candidate: 1,
            steps: 1,
            early_stop_patience: 1,
        };
        let result = infer_params(&[], &[], &config);
        let v = result.to_vec();
        assert_eq!(v.len(), Params::N);
    }

    #[test]
    fn test_preprocess_observations_groups_by_seed() {
        let obs = vec![
            Observation {
                seed_index: 0, vx: 0, vy: 0, vw: 2, vh: 2,
                grid: vec![
                    vec![TerrainType::Plains, TerrainType::Forest],
                    vec![TerrainType::Settlement, TerrainType::Ocean],
                ],
            },
            Observation {
                seed_index: 2, vx: 5, vy: 5, vw: 1, vh: 1,
                grid: vec![vec![TerrainType::Ruin]],
            },
        ];
        let processed = preprocess_observations(&obs);
        assert_eq!(processed.len(), 2); // seed 0 and seed 2
        assert_eq!(processed[0].cells.len(), 4); // 2x2 viewport
        assert_eq!(processed[1].cells.len(), 1); // 1x1 viewport
    }
}
