use std::env;

use anyhow::{Context, Result};

use astar_simulator::io::{
    self, GroundTruthData, QueryResult, ReplayData, RoundDetails,
};
use astar_simulator::inference::{self, InferenceConfig, Observation};
use astar_simulator::montecarlo::{run_montecarlo, run_montecarlo_from_replay, run_ensemble_montecarlo};
use astar_simulator::params::Params;
use astar_simulator::scoring::{score_prediction, competition_score};
use astar_simulator::replay_eval::{self, ReplayInferConfig};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Parse params from JSON — accepts either a struct `{"field": val, ...}`
/// or a legacy flat vector `[val, val, ...]`.
fn parse_params(json: &str) -> Result<Params> {
    // Try struct format first, fall back to vector format.
    serde_json::from_str::<Params>(json)
        .or_else(|_| {
            let v: Vec<f32> = serde_json::from_str(json)
                .context("params JSON is neither a valid Params struct nor a float vector")?;
            Ok(Params::from_vec(&v))
        })
}

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  astar-simulator oracle <round_id> <seed> <n_sims>");
    eprintln!("  astar-simulator montecarlo <round_id> <n_sims> [params_json]");
    eprintln!("  astar-simulator infer <round_id> [budget]");
    eprintln!("  astar-simulator ensemble <round_id> <n_sims> <top_k>");
    eprintln!("  astar-simulator validate-all [n_sims]");
    eprintln!("  astar-simulator replay-eval <round_id> <n_sims> [params_json]");
    eprintln!("  astar-simulator replay-infer <round_id> [sims_per_pair]");
    eprintln!("");
    eprintln!("  params_json: JSON object with named fields, or legacy float array");
    std::process::exit(1);
}

/// Parse a seed_index from a query filename of the form
/// `map_idx=N_...json`.
fn parse_seed_index_from_filename(name: &str) -> Option<usize> {
    // Example: "map_idx=0_run_seed_idx=12_r=24_c=13_w=15_h=15.json"
    let prefix = "map_idx=";
    let start = name.find(prefix)? + prefix.len();
    let rest = &name[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ── Subcommands ───────────────────────────────────────────────────────────────

/// oracle <round_id> <seed> <n_sims> [params_json]
fn cmd_oracle(round_id: &str, seed_index: usize, n_sims: usize, params_json: Option<&str>) -> Result<()> {
    let gt = GroundTruthData::load(round_id, seed_index)
        .with_context(|| format!("loading ground truth for round={round_id} seed={seed_index}"))?;

    let params = match params_json {
        Some(json) => parse_params(json)?,
        None => Params::default_prior(),
    };

    // Try replay-based oracle first (exact initial state), fall back to initial_state
    let prediction = match ReplayData::load(round_id, seed_index) {
        Ok(replay) => run_montecarlo_from_replay(&replay, &params, n_sims, 50, seed_index as u64),
        Err(_) => {
            let details = RoundDetails::load(round_id)?;
            let initial = &details.initial_states[seed_index];
            run_montecarlo(initial, &params, n_sims, 50, seed_index as u64)
        }
    };

    let kl = score_prediction(&prediction, &gt.ground_truth);
    let cs = competition_score(kl);
    println!("kl={kl:.6}  score={cs:.2}/100");
    Ok(())
}

/// montecarlo <round_id> <n_sims> [params_json]
fn cmd_montecarlo(round_id: &str, n_sims: usize, params_json: Option<&str>) -> Result<()> {
    let details = RoundDetails::load(round_id)
        .with_context(|| format!("loading round details for round={round_id}"))?;

    let params = match params_json {
        Some(json) => parse_params(json)?,
        None => Params::default_prior(),
    };

    for seed_index in 0..details.seeds_count {
        let initial_state = &details.initial_states[seed_index];
        let prediction = run_montecarlo(initial_state, &params, n_sims, 50, seed_index as u64);
        let json = serde_json::to_string(&prediction)
            .with_context(|| format!("serialising prediction for seed={seed_index}"))?;
        println!("SEED_{seed_index}: {json}");
    }

    Ok(())
}

/// infer <round_id> [budget]
fn cmd_infer(round_id: &str, budget: usize) -> Result<()> {
    eprintln!("loading round details for {round_id}...");
    let details = RoundDetails::load(round_id)
        .with_context(|| format!("loading round details for round={round_id}"))?;
    eprintln!(
        "  {} initial states, {} seeds",
        details.initial_states.len(),
        details.seeds_count,
    );

    // Scan the query directory for JSON files.
    let query_dir = io::data_dir().join(round_id).join("query");
    let entries = std::fs::read_dir(&query_dir)
        .with_context(|| format!("reading query directory {:?}", query_dir))?;

    eprintln!("loading observations from {:?}...", query_dir);
    let mut observations: Vec<Observation> = Vec::new();

    for entry in entries {
        let entry = entry.with_context(|| format!("iterating query directory {:?}", query_dir))?;
        let path = entry.path();

        // Only process .json files.
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_owned();

        let seed_index = match parse_seed_index_from_filename(&filename) {
            Some(idx) => idx,
            None => {
                eprintln!("warning: could not parse seed index from filename {:?}, skipping", filename);
                continue;
            }
        };

        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading query file {:?}", path))?;
        let qr: QueryResult = serde_json::from_str(&text)
            .with_context(|| format!("parsing query JSON from {:?}", path))?;

        observations.push(Observation {
            seed_index,
            vx: qr.viewport.x,
            vy: qr.viewport.y,
            vw: qr.viewport.w,
            vh: qr.viewport.h,
            grid: qr.grid,
        });
    }

    eprintln!("  loaded {} observations", observations.len());

    let default = InferenceConfig::default();
    let config = InferenceConfig {
        sims_per_candidate: default.sims_per_candidate * budget,
        population_size: default.population_size * budget,
        ..default
    };
    if budget > 1 {
        eprintln!("  inference budget x{budget}: population={}, sims/candidate={}",
                  config.population_size, config.sims_per_candidate);
    }
    let inferred = inference::infer_params(&details.initial_states, &observations, &config);

    let json = serde_json::to_string(&inferred)
        .context("serialising inferred params")?;
    println!("{json}");

    Ok(())
}

/// ensemble <round_id> <n_sims> <top_k>
fn cmd_ensemble(round_id: &str, total_sims: usize, top_k: usize) -> Result<()> {
    let details = RoundDetails::load(round_id)
        .with_context(|| format!("loading round details for round={round_id}"))?;

    // Load query observations for inference
    let query_dir = io::data_dir().join(round_id).join("query");
    let mut observations: Vec<Observation> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&query_dir) {
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
            let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_owned();
            let seed_index = match parse_seed_index_from_filename(&filename) {
                Some(idx) => idx,
                None => continue,
            };
            let text = std::fs::read_to_string(&path)?;
            let qr: QueryResult = serde_json::from_str(&text)?;
            observations.push(Observation {
                seed_index, vx: qr.viewport.x, vy: qr.viewport.y,
                vw: qr.viewport.w, vh: qr.viewport.h, grid: qr.grid,
            });
        }
    }

    // Run ensemble inference
    let config = InferenceConfig::default();
    let top_params = if observations.is_empty() {
        eprintln!("No queries found, using default params only");
        vec![inference::ScoredParams { params: Params::default_prior(), score: 0.0 }]
    } else {
        inference::infer_top_k(&details.initial_states, &observations, &config, top_k)
    };

    let param_sets: Vec<(Params, f64)> = top_params.into_iter()
        .map(|sp| (sp.params, sp.score))
        .collect();

    let sims_per_member = total_sims / param_sets.len();
    eprintln!("Running ensemble MC: {} members × {} sims = {} total per seed",
              param_sets.len(), sims_per_member, param_sets.len() * sims_per_member);

    for seed_index in 0..details.seeds_count {
        let initial = &details.initial_states[seed_index];
        let prediction = run_ensemble_montecarlo(
            initial, &param_sets, sims_per_member, 50, seed_index as u64 * 10000,
        );
        let json = serde_json::to_string(&prediction)
            .with_context(|| format!("serialising prediction for seed={seed_index}"))?;
        println!("SEED_{seed_index}: {json}");
    }

    Ok(())
}

/// replay-eval <round_id> <n_sims> [params_json]
fn cmd_replay_eval(round_id: &str, n_sims: usize, params_json: Option<&str>) -> Result<()> {
    let params = match params_json {
        Some(json) => parse_params(json)?,
        None => Params::default_prior(),
    };

    // Load all replays for this round
    let mut replays = Vec::new();
    for seed in 0..5 {
        match ReplayData::load(round_id, seed) {
            Ok(r) => replays.push(r),
            Err(e) => {
                eprintln!("skipping seed {seed}: {e}");
            }
        }
    }
    if replays.is_empty() {
        anyhow::bail!("no replays found for round {round_id}");
    }

    let mut round_ll = 0.0;
    let mut round_pairs = 0;

    for replay in &replays {
        // Propagate hidden state (tech, longships, damage) through the replay
        let hidden_maps = replay_eval::propagate_hidden_states(replay, &params);

        let mut seed_ll = 0.0;
        for step in 0..replay.frames.len() - 1 {
            let result = replay_eval::eval_step_pair(
                replay, step, &params, &hidden_maps[step], n_sims,
            );
            seed_ll += result.grid_ll;
            round_pairs += 1;

            let json = serde_json::to_string(&result).unwrap();
            println!("PAIR {json}");
        }
        round_ll += seed_ll;
        eprintln!(
            "  seed {} LL={:.2} ({} pairs)",
            replay.seed_index,
            seed_ll,
            replay.frames.len() - 1,
        );
    }

    eprintln!("round total LL={:.2} ({} pairs)", round_ll, round_pairs);
    println!(
        "ROUND_SUMMARY {{\"round_id\":\"{}\",\"total_ll\":{:.6},\"n_pairs\":{}}}",
        round_id, round_ll, round_pairs,
    );

    Ok(())
}

/// replay-infer <round_id> [sims_per_pair]
fn cmd_replay_infer(round_id: &str, sims_per_pair: usize) -> Result<()> {
    let mut replays = Vec::new();
    for seed in 0..5 {
        match ReplayData::load(round_id, seed) {
            Ok(r) => replays.push(r),
            Err(e) => {
                eprintln!("skipping seed {seed}: {e}");
            }
        }
    }
    if replays.is_empty() {
        anyhow::bail!("no replays found for round {round_id}");
    }

    let config = ReplayInferConfig {
        sims_per_pair,
        ..ReplayInferConfig::default()
    };

    let (best_params, best_ll) = replay_eval::infer_replay_params(&replays, &config);

    let params_json =
        serde_json::to_string(&best_params).context("serialising inferred params")?;
    println!("{params_json}");
    eprintln!("best LL={:.2}", best_ll);

    Ok(())
}

/// validate-all [n_sims]
fn cmd_validate_all(n_sims: usize) -> Result<()> {
    let rounds = io::list_rounds_with_analysis()
        .context("listing rounds with analysis data")?;

    let params = Params::default_prior();
    const SEEDS: usize = 5;

    for round_id in &rounds {
        let mut round_total = 0.0_f64;
        let mut round_count = 0_usize;

        for seed_index in 0..SEEDS {
            let replay = match ReplayData::load(round_id, seed_index) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  skipping {round_id} seed={seed_index}: {e}");
                    continue;
                }
            };
            let gt = match GroundTruthData::load(round_id, seed_index) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("  skipping {round_id} seed={seed_index}: {e}");
                    continue;
                }
            };

            let prediction =
                run_montecarlo_from_replay(&replay, &params, n_sims, 50, seed_index as u64);
            let kl = score_prediction(&prediction, &gt.ground_truth);

            let cs = competition_score(kl);
            println!("{round_id}  seed={seed_index}  kl={kl:.6}  score={cs:.2}");
            round_total += kl;
            round_count += 1;
        }

        if round_count > 0 {
            let avg = round_total / round_count as f64;
            let avg_cs = competition_score(avg);
            println!("{round_id}  avg_kl={avg:.6}  avg_score={avg_cs:.2}/100");
        }
    }

    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        usage();
    }

    match args[1].as_str() {
        "oracle" => {
            if args.len() < 5 || args.len() > 6 {
                eprintln!("Usage: astar-simulator oracle <round_id> <seed> <n_sims> [params_json]");
                std::process::exit(1);
            }
            let round_id = &args[2];
            let seed: usize = args[3]
                .parse()
                .with_context(|| format!("parsing seed {:?}", args[3]))?;
            let n_sims: usize = args[4]
                .parse()
                .with_context(|| format!("parsing n_sims {:?}", args[4]))?;
            let params_json = args.get(5).map(|s| s.as_str());
            cmd_oracle(round_id, seed, n_sims, params_json)
        }

        "montecarlo" => {
            if args.len() < 4 || args.len() > 5 {
                eprintln!("Usage: astar-simulator montecarlo <round_id> <n_sims> [params_json]");
                std::process::exit(1);
            }
            let round_id = &args[2];
            let n_sims: usize = args[3]
                .parse()
                .with_context(|| format!("parsing n_sims {:?}", args[3]))?;
            let params_json = args.get(4).map(|s| s.as_str());
            cmd_montecarlo(round_id, n_sims, params_json)
        }

        "infer" => {
            if args.len() < 3 || args.len() > 4 {
                eprintln!("Usage: astar-simulator infer <round_id> [budget]");
                std::process::exit(1);
            }
            let round_id = &args[2];
            let budget: usize = if args.len() >= 4 {
                args[3].parse().context("parsing budget")?
            } else {
                1
            };
            cmd_infer(round_id, budget)
        }

        "ensemble" => {
            if args.len() != 5 {
                eprintln!("Usage: astar-simulator ensemble <round_id> <n_sims> <top_k>");
                std::process::exit(1);
            }
            let round_id = &args[2];
            let n_sims: usize = args[3].parse().context("parsing n_sims")?;
            let top_k: usize = args[4].parse().context("parsing top_k")?;
            cmd_ensemble(round_id, n_sims, top_k)
        }

        "replay-eval" => {
            if args.len() < 4 || args.len() > 5 {
                eprintln!("Usage: astar-simulator replay-eval <round_id> <n_sims> [params_json]");
                std::process::exit(1);
            }
            let round_id = &args[2];
            let n_sims: usize = args[3].parse().context("parsing n_sims")?;
            let params_json = args.get(4).map(|s| s.as_str());
            cmd_replay_eval(round_id, n_sims, params_json)
        }

        "replay-infer" => {
            if args.len() < 3 || args.len() > 4 {
                eprintln!("Usage: astar-simulator replay-infer <round_id> [sims_per_pair]");
                std::process::exit(1);
            }
            let round_id = &args[2];
            let sims_per_pair: usize = if args.len() >= 4 {
                args[3].parse().context("parsing sims_per_pair")?
            } else {
                30
            };
            cmd_replay_infer(round_id, sims_per_pair)
        }

        "validate-all" => {
            let n_sims: usize = if args.len() >= 3 {
                args[2]
                    .parse()
                    .with_context(|| format!("parsing n_sims {:?}", args[2]))?
            } else {
                100
            };
            cmd_validate_all(n_sims)
        }

        _ => {
            eprintln!("Unknown command: {}", args[1]);
            usage();
        }
    }
}
