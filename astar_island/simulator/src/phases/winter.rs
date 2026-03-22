use rand::prelude::*;

use crate::params::Params;
use crate::terrain::TerrainType;
use crate::world::World;

/// Phase 6: Winter.
///
/// Two sub-phases:
///
/// 1. **Food loss + collapse**: Every alive settlement loses food by a
///    stochastic per-step draw.  Settlements below `collapse_threshold` die.
///
/// 2. **Global catastrophe**: Each step draws a catastrophe severity from
///    an exponential distribution.  When the severity exceeds a threshold,
///    each settlement independently dies with a probability proportional to
///    the excess severity.  This models epidemics, harsh winters, and other
///    mass-death events observed in the replay data (20-30% death spikes).
pub fn phase_winter(world: &mut World, params: &Params) {
    let n = world.settlements.len();

    // ── Sub-phase 1: food loss + collapse ────────────────────────────────
    // Stochastic winter severity: i.i.d. uniform draw per step.
    // Uniform(−a, a) has std = a / √3, so a = std × √3.
    let half_width = params.winter_severity * 1.732_f32;
    let severity: f32 = world.rng.random::<f32>() * 2.0 * half_width - half_width;

    let mut to_disperse: Vec<(usize, usize, f32, u16)> = Vec::new();

    for i in 0..n {
        if !world.settlements[i].alive {
            continue;
        }

        world.settlements[i].food =
            (world.settlements[i].food - severity).max(0.0);

        if world.settlements[i].food < params.collapse_threshold {
            let x = world.settlements[i].x;
            let y = world.settlements[i].y;
            let pop = world.settlements[i].population;
            let owner = world.settlements[i].owner_id;

            world.grid[y][x] = TerrainType::Ruin;
            world.settlements[i].alive = false;

            to_disperse.push((x, y, pop, owner));
        }
    }

    // Disperse population from collapsed settlements.
    for (dx, dy, dead_pop, dead_owner) in to_disperse {
        let neighbours: Vec<usize> = world
            .settlements_near(dx, dy, params.winter_dispersal_range)
            .filter(|&i| world.settlements[i].owner_id == dead_owner)
            .collect();

        if neighbours.is_empty() {
            continue;
        }

        let share = dead_pop / neighbours.len() as f32;
        for ni in neighbours {
            world.settlements[ni].population += share;
        }
    }

    // ── Sub-phase 2: global catastrophe ──────────────────────────────────
    // Draw catastrophe severity from exponential distribution (via -ln(U)).
    // When severity > 1, it's a catastrophic event; each settlement dies
    // with probability proportional to (severity - 1) * catastrophe_death_rate.
    let u: f32 = world.rng.random::<f32>().max(1e-10);
    let catastrophe_severity = -u.ln() * params.catastrophe_freq;

    if catastrophe_severity > 1.0 {
        let excess = catastrophe_severity - 1.0;
        let p_death = (params.catastrophe_death_rate * excess).min(0.95);

        let mut catastrophe_disperse: Vec<(usize, usize, f32, u16)> = Vec::new();

        for i in 0..world.settlements.len() {
            if !world.settlements[i].alive {
                continue;
            }

            let roll: f32 = world.rng.random();
            if roll < p_death {
                let x = world.settlements[i].x;
                let y = world.settlements[i].y;
                let pop = world.settlements[i].population;
                let owner = world.settlements[i].owner_id;

                world.grid[y][x] = TerrainType::Ruin;
                world.settlements[i].alive = false;

                catastrophe_disperse.push((x, y, pop, owner));
            }
        }

        // Disperse population from catastrophe deaths.
        for (dx, dy, dead_pop, dead_owner) in catastrophe_disperse {
            let neighbours: Vec<usize> = world
                .settlements_near(dx, dy, params.winter_dispersal_range)
                .filter(|&i| world.settlements[i].owner_id == dead_owner)
                .collect();

            if neighbours.is_empty() {
                continue;
            }

            let share = dead_pop / neighbours.len() as f32;
            for ni in neighbours {
                world.settlements[ni].population += share;
            }
        }
    }
}
