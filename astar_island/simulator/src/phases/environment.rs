use rand::prelude::*;

use crate::params::Params;
use crate::settlement::Settlement;
use crate::terrain::TerrainType;
use crate::world::World;

/// Phase 7: Ruin Transitions.
///
/// Only ruins that existed **before** this step (`pre_step_ruins`) are eligible
/// for transition.  New ruins created during conflict/winter this step are left
/// alone until the next step.
pub fn phase_environment(
    world: &mut World,
    params: &Params,
    pre_step_ruins: &[bool],
) {
    // Collect eligible ruin cells up front so we can mutate the grid/settlements
    // separately.
    let w = world.width;
    let mut ruin_cells: Vec<(usize, usize)> = Vec::new();
    for y in 0..world.height {
        for x in 0..w {
            if pre_step_ruins[y * w + x] && world.grid[y][x] == TerrainType::Ruin {
                ruin_cells.push((x, y));
            }
        }
    }

    let mut new_settlements: Vec<Settlement> = Vec::new();

    for (rx, ry) in ruin_cells {
        // Find nearby alive settlements within reclaim_range.
        let nearby: Vec<usize> = world.settlements_near(rx, ry, params.reclaim_range).collect();
        let n_nearby = nearby.len() as f32;

        let adj_ocean = world.has_adjacent_ocean(rx, ry);

        // Compute transition probabilities.
        // Sublinear scaling: more nearby settlements slightly increase reclaim chance,
        // but the effect saturates quickly to avoid over-reclaiming.
        let density_factor = if n_nearby > 0.0 { n_nearby / (n_nearby + 1.0) } else { 0.0 };
        let raw_settle = params.ruin_to_settlement * density_factor;
        let raw_port = if adj_ocean { raw_settle * 0.1 } else { 0.0 };
        let p_forest = params.ruin_to_forest;
        let total_raw = raw_settle + raw_port + p_forest;
        let (p_settlement, p_port) = if total_raw > 0.95 {
            // Scale settlement & port probabilities to fit within budget
            let scale = (0.95 - p_forest).max(0.0) / (raw_settle + raw_port).max(1e-6);
            (raw_settle * scale, raw_port * scale)
        } else {
            (raw_settle, raw_port)
        };

        // Sample transition.
        let roll: f32 = world.rng.random();

        if roll < p_settlement {
            // Spawn as regular settlement.
            let (nearest_food, random_owner) = reclaim_stats(world, &nearby);
            let new_s = Settlement {
                x: rx,
                y: ry,
                population: 0.4,
                defense: 0.15,
                food: nearest_food * 0.2,
                wealth: 0.0,
                has_port: false,
                alive: true,
                owner_id: random_owner,
                tech_level: 0.0,
                longships: 0,
                total_damage: 0.0,
            };
            world.grid[ry][rx] = TerrainType::Settlement;
            new_settlements.push(new_s);
        } else if roll < p_settlement + p_port {
            // Spawn as port.
            let (nearest_food, random_owner) = reclaim_stats(world, &nearby);
            let new_s = Settlement {
                x: rx,
                y: ry,
                population: 0.4,
                defense: 0.15,
                food: nearest_food * 0.2,
                wealth: 0.0,
                has_port: true,
                alive: true,
                owner_id: random_owner,
                tech_level: 0.0,
                longships: 0,
                total_damage: 0.0,
            };
            world.grid[ry][rx] = TerrainType::Port;
            new_settlements.push(new_s);
        } else if roll < p_settlement + p_port + p_forest {
            world.grid[ry][rx] = TerrainType::Forest;
        } else {
            // Transition to Plains (do nothing if roll falls in [p_plains, 1.0]
            // because p_plains = 1 - others; this branch is "stay plains" or
            // become plains from ruin).
            world.grid[ry][rx] = TerrainType::Plains;
        }
    }

    world.settlements.extend(new_settlements);

    world.update_settlement_grid();
}

/// Return (food, owner_id) from a randomly chosen nearby settlement.
fn reclaim_stats(world: &mut World, nearby: &[usize]) -> (f32, u16) {
    if nearby.is_empty() {
        return (0.0, 0);
    }
    let s = &world.settlements[*nearby.choose(&mut world.rng).unwrap()];
    (s.food, s.owner_id)
}
