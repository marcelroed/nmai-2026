use astar_simulator::io::{list_rounds_with_analysis, GroundTruthData, ReplayData};
use astar_simulator::montecarlo::run_montecarlo_from_replay;
use astar_simulator::params::Params;
use astar_simulator::scoring::score_prediction;

/// Load a replay and the corresponding ground truth for the first available
/// round (seed 0).  Run 100 Monte Carlo simulations from replay frame 0 and
/// score the resulting prediction against the ground truth.
///
/// Assertions:
///   - The KL divergence is finite.
///   - The KL divergence is non-negative.
#[test]
fn test_oracle_montecarlo_vs_ground_truth() {
    let rounds = list_rounds_with_analysis().expect("list rounds");
    assert!(!rounds.is_empty(), "no rounds with analysis data found");

    let round_id = &rounds[0];

    let replay = ReplayData::load(round_id, 0).expect("load replay");
    let gt = GroundTruthData::load(round_id, 0).expect("load ground truth");

    let params = Params::default_prior();

    // Use steps = number of replay frames - 1, capped at 50 for speed.
    let steps = ((replay.frames.len().saturating_sub(1)) as u32).min(50);

    let prediction = run_montecarlo_from_replay(&replay, &params, 100, steps, 0);

    let kl = score_prediction(&prediction, &gt.ground_truth);

    println!(
        "Oracle KL divergence (round={}, {} sims, {} steps): {:.6}",
        round_id, 100, steps, kl
    );

    assert!(kl.is_finite(), "KL divergence must be finite, got {}", kl);
    assert!(kl >= 0.0, "KL divergence must be non-negative, got {}", kl);
}
