pub mod conflict;
pub mod environment;
pub mod food;
pub mod growth;
pub mod pop_defense;
pub mod trade;
pub mod winter;

use crate::params::Params;
use crate::terrain::TerrainType;
use crate::world::World;

/// Run one complete simulation step (one year).
pub fn step(world: &mut World, params: &Params) {
    // Track ruins that existed BEFORE this step as a flat bool grid (no hashing).
    let w = world.width;
    let h = world.height;
    let mut pre_step_ruins = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if world.grid[y][x] == TerrainType::Ruin {
                pre_step_ruins[y * w + x] = true;
            }
        }
    }

    food::phase_food(world, params);
    pop_defense::phase_pop_defense(world, params);
    growth::phase_growth(world, params);
    conflict::phase_conflict(world, params);
    trade::phase_trade(world, params);
    winter::phase_winter(world, params);
    environment::phase_environment(world, params, &pre_step_ruins);
}

/// Run the full simulation for the given number of steps.
pub fn simulate(world: &mut World, params: &Params, steps: u32) {
    for _ in 0..steps {
        step(world, params);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ReplayData;
    use crate::world::World;

    const ROUND_ID: &str = "ae78003a-4efe-425a-881a-d16a39bca0ad";

    /// Load replay frame 0, run 1 step, verify basic invariants.
    #[test]
    fn test_step_no_panic_and_basic_invariants() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let mut world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        // Use params whose collapse_threshold is below the max food value (0.998)
        // so settlements survive at least one winter step.
        let mut params = Params::default_prior();
        // Food is clamped to [0, 0.998]; collapse_threshold must be < food values
        // seen in the replay (food ~ 0.6–0.8) to avoid killing everything in one step.
        params.collapse_threshold = 0.0;
        params.winter_severity = 0.01;

        let orig_h = world.height;
        let orig_w = world.width;

        // Run 1 step — must not panic.
        step(&mut world, &params);

        // Grid dimensions must be unchanged.
        assert_eq!(world.height, orig_h);
        assert_eq!(world.width, orig_w);
        assert_eq!(world.grid.len(), orig_h);
        assert_eq!(world.grid[0].len(), orig_w);

        // Settlements still exist in the vector (some may have died but the
        // vec is non-empty and at least one alive settlement remains).
        assert!(!world.settlements.is_empty());
        assert!(world.settlements.iter().any(|s| s.alive));
    }

    /// Run 50 steps to exercise the ruin cycle and all phases.
    #[test]
    fn test_simulate_50_steps_no_panic() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let mut world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);
        let params = Params::default_prior();

        // Must not panic over 50 steps.
        simulate(&mut world, &params, 50);

        // Grid dimensions still unchanged.
        assert_eq!(world.grid.len(), replay.height);
        assert_eq!(world.grid[0].len(), replay.width);
    }
}
