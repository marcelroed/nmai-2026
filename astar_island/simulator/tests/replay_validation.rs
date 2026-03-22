use astar_simulator::io::{list_rounds_with_analysis, ReplayData};
use astar_simulator::phases::{self, simulate};
use astar_simulator::{Params, World};

/// Load a replay, create a World from frame 0, run exactly 1 step, and
/// compare the resulting prediction snapshot against the replay's frame 1.
///
/// We print the match percentage (expect > 60% with default params) and assert
/// structural invariants.
#[test]
fn test_single_step_replay_fidelity() {
    let rounds = list_rounds_with_analysis().expect("list rounds");
    assert!(!rounds.is_empty(), "no rounds with analysis data found");

    let round_id = &rounds[0];
    let replay = ReplayData::load(round_id, 0).expect("load replay");

    // Need at least 2 frames to compare step 0 → step 1.
    if replay.frames.len() < 2 {
        eprintln!("replay has fewer than 2 frames, skipping fidelity check");
        return;
    }

    let frame0 = &replay.frames[0];
    let frame1 = &replay.frames[1];

    let mut world = World::from_replay_frame(frame0, replay.width, replay.height, replay.sim_seed);
    let params = Params::default_prior();

    // Run exactly 1 simulation step.
    phases::step(&mut world, &params);

    let snapshot = world.prediction_snapshot();

    // Count how many cells match the ground-truth frame 1 grid.
    let mut matching = 0usize;
    let mut total = 0usize;
    for y in 0..replay.height {
        for x in 0..replay.width {
            let sim_class = snapshot[y][x];
            let gt_class = frame1.grid[y][x].prediction_class();
            total += 1;
            if sim_class == gt_class {
                matching += 1;
            }
        }
    }

    let match_pct = (matching as f64 / total as f64) * 100.0;
    println!(
        "Single-step fidelity: {}/{} cells match ({:.1}%)",
        matching, total, match_pct
    );

    // Structural assertions.
    // Grid dimensions must be unchanged.
    assert_eq!(snapshot.len(), replay.height, "height mismatch");
    assert_eq!(snapshot[0].len(), replay.width, "width mismatch");

    // At least some settlements must still be alive in the simulated world.
    let alive = world.settlements.iter().filter(|s| s.alive).count();
    assert!(alive > 0, "expected at least one alive settlement after 1 step");
}

/// Load a replay, create a World from frame 0, run 50 steps.
///
/// Primary assertion: no panics occur and at least one settlement survives.
#[test]
fn test_full_simulation_runs_without_panic() {
    let rounds = list_rounds_with_analysis().expect("list rounds");
    assert!(!rounds.is_empty(), "no rounds with analysis data found");

    let round_id = &rounds[0];
    let replay = ReplayData::load(round_id, 0).expect("load replay");
    let frame0 = &replay.frames[0];

    let mut world = World::from_replay_frame(frame0, replay.width, replay.height, replay.sim_seed);
    let params = Params::default_prior();

    // Must not panic.
    simulate(&mut world, &params, 50);

    // Grid dimensions unchanged.
    assert_eq!(world.grid.len(), replay.height);
    assert_eq!(world.grid[0].len(), replay.width);

    // At least one settlement alive.
    let alive = world.settlements.iter().filter(|s| s.alive).count();
    assert!(alive > 0, "expected at least one alive settlement after 50 steps");
}
