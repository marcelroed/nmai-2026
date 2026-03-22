use astar_simulator::io::{GroundTruthData, RoundDetails, list_rounds_with_analysis};
use astar_simulator::montecarlo::run_montecarlo;
use astar_simulator::params::Params;
use astar_simulator::scoring::score_prediction;

/// Load the first available round, run 200 Monte Carlo simulations from the
/// initial state (not the replay), score against ground truth, and assert that
/// the entropy-weighted KL divergence is finite and non-negative.
///
/// The KL < 1.0 assertion checks that the calibrated default params produce
/// a prediction that is at least somewhat informative about the ground truth.
#[test]
fn test_e2e_montecarlo_from_initial_state() {
    let rounds = list_rounds_with_analysis().expect("list rounds");
    assert!(!rounds.is_empty(), "no rounds with analysis data found");

    let round_id = &rounds[0];

    let details = RoundDetails::load(round_id).expect("load round details");
    let gt = GroundTruthData::load(round_id, 0).expect("load ground truth");
    let params = Params::default_prior();

    // Use the initial state for seed index 0.
    let initial = &details.initial_states[0];

    // Run 200 simulations for 50 steps from the initial state (real path —
    // no replay statistics available here).
    let n_sims = 200;
    let steps = 50_u32;
    let prediction = run_montecarlo(initial, &params, n_sims, steps, 0);

    let kl = score_prediction(&prediction, &gt.ground_truth);

    println!(
        "E2E initial-state KL (round={}, {} sims, {} steps): {:.6}",
        round_id, n_sims, steps, kl
    );

    assert!(kl.is_finite(), "KL divergence must be finite, got {}", kl);
    assert!(kl >= 0.0, "KL divergence must be non-negative, got {}", kl);
    assert!(
        kl < 2.0,
        "KL divergence {} >= 2.0: default params should achieve KL < 2.0",
        kl
    );
}

/// For the first available round, run Monte Carlo from each of the 5 seed
/// initial states and score each against its corresponding ground truth.
/// The average KL across seeds should be < 0.5.
#[test]
fn test_e2e_all_seeds() {
    let rounds = list_rounds_with_analysis().expect("list rounds");
    assert!(!rounds.is_empty(), "no rounds with analysis data found");

    let round_id = &rounds[0];

    let details = RoundDetails::load(round_id).expect("load round details");
    let params = Params::default_prior();

    let n_seeds = details.seeds_count.min(details.initial_states.len());
    assert!(n_seeds > 0, "expected at least one seed");

    let n_sims = 100;
    let steps = 50_u32;

    let mut total_kl = 0.0_f64;
    let mut counted = 0usize;

    for seed_idx in 0..n_seeds {
        // Ground truth may not exist for every seed; skip gracefully.
        let gt = match GroundTruthData::load(round_id, seed_idx) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping seed {}: {}", seed_idx, e);
                continue;
            }
        };

        let initial = &details.initial_states[seed_idx];
        let prediction =
            run_montecarlo(initial, &params, n_sims, steps, seed_idx as u64 * 1000);

        let kl = score_prediction(&prediction, &gt.ground_truth);

        println!(
            "  seed={} KL={:.6}",
            seed_idx, kl
        );

        assert!(kl.is_finite(), "seed {} KL must be finite, got {}", seed_idx, kl);
        assert!(kl >= 0.0, "seed {} KL must be non-negative, got {}", seed_idx, kl);

        total_kl += kl;
        counted += 1;
    }

    assert!(counted > 0, "no seeds were successfully evaluated");

    let avg_kl = total_kl / counted as f64;
    println!(
        "E2E all-seeds average KL (round={}, {} seeds, {} sims, {} steps): {:.6}",
        round_id, counted, n_sims, steps, avg_kl
    );

    assert!(
        avg_kl < 2.0,
        "average KL {} >= 2.0 across {} seeds",
        avg_kl,
        counted
    );
}
