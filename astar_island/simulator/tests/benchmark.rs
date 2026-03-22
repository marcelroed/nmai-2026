use std::time::Instant;

use astar_simulator::io::{list_rounds_with_analysis, RoundDetails};
use astar_simulator::params::Params;
use astar_simulator::phases::simulate;
use astar_simulator::World;

/// Time 1000 simulations of 50 steps each, starting from the initial state of
/// the first available round.  Prints simulations/second.  This is an
/// informational benchmark — it does not assert a specific throughput target.
#[test]
fn test_simulation_throughput() {
    let rounds = list_rounds_with_analysis().expect("list rounds");
    assert!(!rounds.is_empty(), "no rounds with analysis data found");

    let round_id = &rounds[0];
    let details = RoundDetails::load(round_id).expect("load round details");
    let params = Params::default_prior();
    let initial = &details.initial_states[0];

    let n_sims = 1000_u64;
    let steps = 50_u32;

    let start = Instant::now();

    for i in 0..n_sims {
        let mut world = World::from_initial_state(initial, &params, i);
        simulate(&mut world, &params, steps);
    }

    let elapsed = start.elapsed();
    let sims_per_sec = n_sims as f64 / elapsed.as_secs_f64();

    println!(
        "Throughput: {:.0} sims/s ({} sims × {} steps in {:.2}s)",
        sims_per_sec,
        n_sims,
        steps,
        elapsed.as_secs_f64()
    );
}
