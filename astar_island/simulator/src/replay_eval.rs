//! Single-step replay evaluation: score simulator params against known rollouts.
//!
//! For each (seed, step_i → step_{i+1}) pair in the replay data, we:
//!   1. Construct a World from frame_i (exact settlement stats from replay).
//!   2. Inject propagated hidden state (tech, longships, total_damage) from
//!      a teacher-forced forward simulation.
//!   3. Run N single-step simulations with different RNG seeds.
//!   4. Compare the distribution of simulated grids against frame_{i+1}.
//!   5. Compute log-likelihood: LL = Σ_cells log P(observed terrain type).

use std::collections::HashMap;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;

use crate::io::ReplayData;
use crate::params::Params;
use crate::phases;
use crate::world::World;

// ── Output types ─────────────────────────────────────────────────────────────

/// Per-cell diagnostic for the worst-fitting cells in a step pair.
#[derive(Debug, Serialize)]
pub struct CellDiag {
    pub y: usize,
    pub x: usize,
    pub from_class: usize,
    pub expected_class: usize,
    pub sim_probs: [f64; 6],
    pub cell_ll: f64,
}

/// Terrain transition mismatch between real and simulated.
#[derive(Debug, Serialize)]
pub struct TransitionDiag {
    pub from_class: usize,
    pub to_class: usize,
    pub real_count: u32,
    pub sim_avg: f64,
}

/// Full result for one step pair evaluation.
#[derive(Debug, Serialize)]
pub struct StepPairResult {
    pub seed_index: usize,
    pub step: u32,
    pub grid_ll: f64,
    pub n_dynamic: usize,
    pub n_changed: usize,
    pub modal_acc: f64,
    pub worst_cells: Vec<CellDiag>,
    pub top_transitions: Vec<TransitionDiag>,
    pub settlement_mae: [f64; 4], // [pop, food, wealth, defense]
}

// ── Per-settlement hidden state ─────────────────────────────────────────────

/// Hidden state for a single settlement (not visible in replay data).
#[derive(Clone, Copy, Default)]
pub struct PerSettlementHidden {
    pub tech_level: f32,
    pub longships: u32,
    pub total_damage: f32,
}

/// Map from (x, y) position to hidden state for that settlement.
pub type HiddenMap = HashMap<(usize, usize), PerSettlementHidden>;

/// Propagate hidden state through a replay via teacher-forced simulation.
///
/// Runs N_PROP simulations per step and averages the hidden state (tech,
/// damage) to get smooth estimates.  At each step:
///   1. Create N_PROP Worlds from the replay frame (visible state).
///   2. Inject hidden state propagated from the previous step.
///   3. Simulate one step per world.
///   4. Average the hidden state across simulations.
///
/// Returns one HiddenMap per frame (frame 0 is always empty/defaults).
pub fn propagate_hidden_states(
    replay: &ReplayData,
    params: &Params,
) -> Vec<HiddenMap> {
    const N_PROP: usize = 32;

    let n_frames = replay.frames.len();
    let mut maps: Vec<HiddenMap> = Vec::with_capacity(n_frames);

    // Frame 0: no prior hidden state
    maps.push(HiddenMap::new());

    for step in 0..n_frames - 1 {
        // Run N_PROP sims and accumulate hidden state per settlement position.
        // Key: (x, y) → (tech_sum, longship_sum, damage_sum, count)
        let mut accum: HashMap<(usize, usize), (f32, f32, f32, u32)> = HashMap::new();

        for i in 0..N_PROP {
            let seed = replay.seed_index as u64 * 1_000_000
                + step as u64 * 10_000
                + 900 + i as u64;
            let mut world = World::from_replay_frame(
                &replay.frames[step],
                replay.width,
                replay.height,
                seed,
            );
            apply_hidden_map(&mut world, &maps[step]);

            phases::step(&mut world, params);

            for s in &world.settlements {
                if s.alive {
                    let entry = accum.entry((s.x, s.y)).or_insert((0.0, 0.0, 0.0, 0));
                    entry.0 += s.tech_level;
                    entry.1 += s.longships as f32;
                    entry.2 += s.total_damage;
                    entry.3 += 1;
                }
            }
        }

        // Average into a HiddenMap
        let mut hidden = HiddenMap::new();
        for ((x, y), (tech_sum, ls_sum, dmg_sum, count)) in &accum {
            let n = *count as f32;
            hidden.insert(
                (*x, *y),
                PerSettlementHidden {
                    tech_level: tech_sum / n,
                    longships: if ls_sum / n > 0.5 { 1 } else { 0 },
                    total_damage: dmg_sum / n,
                },
            );
        }
        maps.push(hidden);
    }

    maps
}

/// Apply per-settlement hidden state to a freshly created World.
/// Settlements not in the map keep their defaults (tech=0, longships=0, damage=0).
fn apply_hidden_map(world: &mut World, map: &HiddenMap) {
    for s in &mut world.settlements {
        if let Some(h) = map.get(&(s.x, s.y)) {
            s.tech_level = h.tech_level;
            s.longships = h.longships;
            s.total_damage = h.total_damage;
        }
    }
}

// ── Core simulation accumulation ─────────────────────────────────────────────

/// Run N single-step simulations from a replay frame, accumulate per-cell
/// terrain class counts. Returns a flat Vec of [u32; 6] indexed by y*width+x.
fn accumulate_grid_counts(
    frame: &crate::io::ReplayFrame,
    width: usize,
    height: usize,
    params: &Params,
    hidden_map: &HiddenMap,
    n_sims: usize,
    base_seed: u64,
) -> Vec<[u32; 6]> {
    (0..n_sims)
        .into_par_iter()
        .fold(
            || vec![[0u32; 6]; width * height],
            |mut counts, i| {
                let seed = base_seed + i as u64;
                let mut world = World::from_replay_frame(frame, width, height, seed);
                apply_hidden_map(&mut world, hidden_map);
                phases::step(&mut world, params);
                for y in 0..height {
                    for x in 0..width {
                        counts[y * width + x][world.grid[y][x].prediction_class()] += 1;
                    }
                }
                counts
            },
        )
        .reduce(
            || vec![[0u32; 6]; width * height],
            |mut a, b| {
                for i in 0..a.len() {
                    for c in 0..6 {
                        a[i][c] += b[i][c];
                    }
                }
                a
            },
        )
}

/// Compute grid log-likelihood from accumulated counts vs the actual next frame.
fn grid_ll_from_counts(
    counts: &[[u32; 6]],
    frame_before: &crate::io::ReplayFrame,
    frame_after: &crate::io::ReplayFrame,
    width: usize,
    height: usize,
    n_sims: usize,
) -> f64 {
    let n_f = n_sims as f64;
    let floor = 0.5 / n_f; // avoid log(0)
    let mut ll = 0.0;
    for y in 0..height {
        for x in 0..width {
            if frame_before.grid[y][x].is_static() {
                continue;
            }
            let to_class = frame_after.grid[y][x].prediction_class();
            let count = counts[y * width + x][to_class];
            let prob = (count as f64 / n_f).max(floor);
            ll += prob.ln();
        }
    }
    ll
}

// ── Fast LL computation (for inference) ──────────────────────────────────────

/// Compute grid LL for one step pair (fast path, no diagnostics).
pub fn step_pair_ll(
    replay: &ReplayData,
    step: usize,
    params: &Params,
    hidden_map: &HiddenMap,
    n_sims: usize,
) -> f64 {
    let base_seed = replay.seed_index as u64 * 1_000_000 + step as u64 * 10_000;
    let counts = accumulate_grid_counts(
        &replay.frames[step],
        replay.width,
        replay.height,
        params,
        hidden_map,
        n_sims,
        base_seed,
    );
    grid_ll_from_counts(
        &counts,
        &replay.frames[step],
        &replay.frames[step + 1],
        replay.width,
        replay.height,
        n_sims,
    )
}

/// Compute total grid LL across all step pairs in all replays.
/// Uses default (empty) hidden state for speed (inference fitness function).
pub fn round_total_ll(replays: &[ReplayData], params: &Params, n_sims: usize) -> f64 {
    let empty = HiddenMap::new();
    replays
        .iter()
        .map(|replay| {
            (0..replay.frames.len() - 1)
                .map(|step| step_pair_ll(replay, step, params, &empty, n_sims))
                .sum::<f64>()
        })
        .sum()
}

// ── Full evaluation with diagnostics ─────────────────────────────────────────

/// Evaluate one step pair with full diagnostics.
pub fn eval_step_pair(
    replay: &ReplayData,
    step: usize,
    params: &Params,
    hidden_map: &HiddenMap,
    n_sims: usize,
) -> StepPairResult {
    let width = replay.width;
    let height = replay.height;
    let seed_index = replay.seed_index;
    let frame_before = &replay.frames[step];
    let frame_after = &replay.frames[step + 1];

    // Extract positions of alive settlements in frame_after
    let after_positions: Vec<(usize, usize, f64, f64, f64, f64)> = frame_after
        .settlements
        .iter()
        .filter(|s| s.alive)
        .map(|s| (s.x, s.y, s.population, s.food, s.wealth, s.defense))
        .collect();
    let n_after = after_positions.len();

    let base_seed = seed_index as u64 * 1_000_000 + step as u64 * 10_000;

    // Combined fold: grid counts + settlement stats
    // Settlement stats: [pop_sum, food_sum, wealth_sum, defense_sum, count]
    type Acc = (Vec<[u32; 6]>, Vec<[f64; 5]>);

    let (grid_counts, settle_stats) = (0..n_sims)
        .into_par_iter()
        .fold(
            || -> Acc {
                (
                    vec![[0u32; 6]; width * height],
                    vec![[0.0f64; 5]; n_after],
                )
            },
            |(mut gc, mut ss), i| {
                let seed = base_seed + i as u64;
                let mut world = World::from_replay_frame(frame_before, width, height, seed);
                apply_hidden_map(&mut world, hidden_map);
                phases::step(&mut world, &params);

                // Grid counts
                for y in 0..height {
                    for x in 0..width {
                        gc[y * width + x][world.grid[y][x].prediction_class()] += 1;
                    }
                }

                // Settlement stats
                for (pi, &(px, py, _, _, _, _)) in after_positions.iter().enumerate() {
                    if let Some(idx) = world.settlement_at(px, py) {
                        let s = &world.settlements[idx];
                        ss[pi][0] += s.population as f64;
                        ss[pi][1] += s.food as f64;
                        ss[pi][2] += s.wealth as f64;
                        ss[pi][3] += s.defense as f64;
                        ss[pi][4] += 1.0;
                    }
                }

                (gc, ss)
            },
        )
        .reduce(
            || -> Acc {
                (
                    vec![[0u32; 6]; width * height],
                    vec![[0.0f64; 5]; n_after],
                )
            },
            |(mut ga, mut sa), (gb, sb)| {
                for i in 0..ga.len() {
                    for c in 0..6 {
                        ga[i][c] += gb[i][c];
                    }
                }
                for i in 0..sa.len() {
                    for j in 0..5 {
                        sa[i][j] += sb[i][j];
                    }
                }
                (ga, sa)
            },
        );

    // Compute grid LL and diagnostics
    let n_f = n_sims as f64;
    let floor = 0.5 / n_f;
    let mut grid_ll = 0.0;
    let mut n_dynamic = 0;
    let mut n_changed = 0;
    let mut n_modal_correct = 0;
    // (y, x, cell_ll, from_class, to_class, probs)
    let mut cell_lls: Vec<(usize, usize, f64, usize, usize, [f64; 6])> = Vec::new();
    let mut real_trans = [[0u32; 6]; 6];
    let mut sim_trans = [[0.0f64; 6]; 6];

    for y in 0..height {
        for x in 0..width {
            let from_class = frame_before.grid[y][x].prediction_class();
            let to_class = frame_after.grid[y][x].prediction_class();
            let c_arr = &grid_counts[y * width + x];

            // Transition tracking (all cells)
            real_trans[from_class][to_class] += 1;
            for c in 0..6 {
                sim_trans[from_class][c] += c_arr[c] as f64 / n_f;
            }

            // Skip static cells for LL
            if frame_before.grid[y][x].is_static() {
                continue;
            }

            n_dynamic += 1;
            if from_class != to_class {
                n_changed += 1;
            }

            let count = c_arr[to_class];
            let prob = (count as f64 / n_f).max(floor);
            let cell_ll = prob.ln();
            grid_ll += cell_ll;

            let modal = c_arr
                .iter()
                .enumerate()
                .max_by_key(|&(_, c)| *c)
                .map(|(c, _)| c)
                .unwrap_or(0);
            if modal == to_class {
                n_modal_correct += 1;
            }

            let probs: [f64; 6] = std::array::from_fn(|c| c_arr[c] as f64 / n_f);
            cell_lls.push((y, x, cell_ll, from_class, to_class, probs));
        }
    }

    // Worst cells (lowest LL)
    cell_lls.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let worst_cells: Vec<CellDiag> = cell_lls
        .iter()
        .take(5)
        .map(|&(y, x, ll, fc, tc, probs)| CellDiag {
            y,
            x,
            from_class: fc,
            expected_class: tc,
            sim_probs: probs,
            cell_ll: ll,
        })
        .collect();

    // Transition mismatches (sorted by absolute delta)
    let mut transitions = Vec::new();
    for from in 0..6 {
        for to in 0..6 {
            let real = real_trans[from][to];
            let sim = sim_trans[from][to];
            let delta = (real as f64 - sim).abs();
            if delta > 0.5 {
                transitions.push(TransitionDiag {
                    from_class: from,
                    to_class: to,
                    real_count: real,
                    sim_avg: sim,
                });
            }
        }
    }
    transitions.sort_by(|a, b| {
        let da = (a.real_count as f64 - a.sim_avg).abs();
        let db = (b.real_count as f64 - b.sim_avg).abs();
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
    transitions.truncate(10);

    // Settlement stat MAE
    let mut settlement_mae = [0.0f64; 4];
    let mut n_matched = 0;
    for (pi, &(_, _, real_pop, real_food, real_wealth, real_def)) in
        after_positions.iter().enumerate()
    {
        let ss = &settle_stats[pi];
        if ss[4] > 0.0 {
            let cnt = ss[4];
            settlement_mae[0] += (real_pop - ss[0] / cnt).abs();
            settlement_mae[1] += (real_food - ss[1] / cnt).abs();
            settlement_mae[2] += (real_wealth - ss[2] / cnt).abs();
            settlement_mae[3] += (real_def - ss[3] / cnt).abs();
            n_matched += 1;
        }
    }
    if n_matched > 0 {
        for i in 0..4 {
            settlement_mae[i] /= n_matched as f64;
        }
    }

    let modal_acc = if n_dynamic > 0 {
        n_modal_correct as f64 / n_dynamic as f64 * 100.0
    } else {
        100.0
    };

    StepPairResult {
        seed_index,
        step: step as u32,
        grid_ll,
        n_dynamic,
        n_changed,
        modal_acc,
        worst_cells,
        top_transitions: transitions,
        settlement_mae,
    }
}

// ── CMA-ES inference for replay-based params ─────────────────────────────────

pub struct ReplayInferConfig {
    pub n_generations: usize,
    pub population_size: usize,
    pub sims_per_pair: usize,
    pub early_stop_patience: usize,
}

impl Default for ReplayInferConfig {
    fn default() -> Self {
        ReplayInferConfig {
            n_generations: 80,
            population_size: 20,
            sims_per_pair: 30,
            early_stop_patience: 15,
        }
    }
}

#[inline]
fn box_muller(u1: f32, u2: f32) -> f32 {
    let u1 = u1.max(1e-10);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Run CMA-ES to find params that maximize total step-pair grid LL.
/// Returns (best_params, best_ll).
pub fn infer_replay_params(
    replays: &[ReplayData],
    config: &ReplayInferConfig,
) -> (Params, f64) {
    use std::time::Instant;

    let lower = Params::lower_bounds();
    let upper = Params::upper_bounds();
    let n_dims = lower.len();

    let total_pairs: usize = replays.iter().map(|r| r.frames.len() - 1).sum();
    let total_sims_per_eval = total_pairs * config.sims_per_pair;

    eprintln!("── replay inference ───────────────────────────────────────");
    eprintln!(
        "  replays: {}  step pairs: {}  sims/pair: {}",
        replays.len(),
        total_pairs,
        config.sims_per_pair,
    );
    eprintln!(
        "  generations: {}  population: {}  sims/eval: {}",
        config.n_generations, config.population_size, total_sims_per_eval,
    );
    eprintln!("───────────────────────────────────────────────────────────");

    let mut mean: Vec<f32> = Params::default_prior().to_vec();
    let mut sigma: Vec<f32> = (0..n_dims)
        .map(|i| 0.10 * (upper[i] - lower[i]).abs())
        .collect();

    let mut best_params = mean.clone();
    let t_start = Instant::now();

    eprintln!("  evaluating initial prior...");
    let mut best_score =
        round_total_ll(replays, &Params::from_vec(&best_params), config.sims_per_pair);
    eprintln!(
        "  initial prior ll={:.2}  ({:.1}s)",
        best_score,
        t_start.elapsed().as_secs_f64()
    );

    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF_CAFE_BABE);

    let mut last_improvement_gen: usize = 0;
    let mut n_restarts = 0u32;
    const MAX_RESTARTS: u32 = 2;

    for generation in 0..config.n_generations {
        let t_gen = Instant::now();

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
                round_total_ll(replays, &p, config.sims_per_pair)
            })
            .collect();

        let mut indexed: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
            let _stale = generation - last_improvement_gen;

            let avg_sigma: f32 = sigma.iter().sum::<f32>() / sigma.len() as f32;
            let marker = if improved { " *" } else { "" };

            eprintln!(
                "gen {:>3}/{}  best={:.2}  gen={:.2}  σ={:.4}  [{:.1}s, {:.1}s/gen, eta {:.0}s]{}",
                gens_done,
                config.n_generations,
                best_score,
                best_gen_score,
                avg_sigma,
                gen_elapsed,
                avg_per_gen,
                eta,
                marker,
            );
        }

        // Early stopping with warm restart
        let stale_generations = generation.saturating_sub(last_improvement_gen);
        if stale_generations >= config.early_stop_patience {
            if n_restarts < MAX_RESTARTS {
                n_restarts += 1;
                eprintln!(
                    "Warm restart #{} at generation {}: resetting sigma",
                    n_restarts, generation
                );
                mean = best_params.clone();
                sigma = (0..n_dims)
                    .map(|i| 0.08 * (upper[i] - lower[i]).abs())
                    .collect();
                last_improvement_gen = generation;
                continue;
            } else {
                eprintln!(
                    "Early stop at generation {generation}: no improvement for {stale_generations} generations"
                );
                break;
            }
        }

        // Elite selection (top 25%)
        let n_elites = ((config.population_size as f64 * 0.25).ceil() as usize).max(1);

        let mut new_mean = vec![0.0_f32; n_dims];
        for &(ei, _) in &indexed[..n_elites] {
            for d in 0..n_dims {
                new_mean[d] += candidates[ei][d];
            }
        }
        let n_f = n_elites as f32;
        for d in 0..n_dims {
            new_mean[d] /= n_f;
        }

        let mut new_sigma = vec![0.0_f32; n_dims];
        for &(ei, _) in &indexed[..n_elites] {
            for d in 0..n_dims {
                let diff = candidates[ei][d] - new_mean[d];
                new_sigma[d] += diff * diff;
            }
        }
        for d in 0..n_dims {
            new_sigma[d] = (new_sigma[d] / n_f).sqrt().max(0.001);
        }

        mean = new_mean;
        sigma = new_sigma;
    }

    let total_elapsed = t_start.elapsed().as_secs_f64();
    eprintln!("───────────────────────────────────────────────────────────");
    eprintln!(
        "  done in {:.1}s  best_ll={:.2}  last improvement at gen {}",
        total_elapsed,
        best_score,
        last_improvement_gen + 1,
    );
    eprintln!("───────────────────────────────────────────────────────────");

    (Params::from_vec(&best_params), best_score)
}
