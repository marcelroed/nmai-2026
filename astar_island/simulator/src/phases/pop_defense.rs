use rand::prelude::*;

use crate::params::Params;
use crate::terrain::TerrainType;
use crate::world::World;

/// Phase 2: Population, Defense, Port Upgrades & Longship Building.
///
/// - Population: logistic growth driven by food, wealth, and current pop.
/// - Defense: recovery proportional to current population.
/// - Port upgrade: coastal settlements may become ports (prob ∝ pop·food·wealth).
/// - Longships: ports may build longships by spending wealth.
pub fn phase_pop_defense(world: &mut World, params: &Params) {
    let n = world.settlements.len();

    for i in 0..n {
        if !world.settlements[i].alive {
            continue;
        }

        // ── Logistic population growth (now includes wealth term) ────────────
        let pop = world.settlements[i].population;
        let food = world.settlements[i].food;
        let wealth = world.settlements[i].wealth;

        let pop_delta = params.pop_growth_rate * food * (1.0 - pop / params.pop_max)
            + params.pop_wealth_coeff * wealth * pop;
        world.settlements[i].population = (pop + pop_delta).max(0.01);

        // ── Starvation: population decline when food is very low ─────────────
        if food < params.collapse_threshold {
            let starvation_loss = params.starvation_rate * pop;
            world.settlements[i].population = (world.settlements[i].population - starvation_loss).max(0.01);

            // Severe starvation can kill established settlements.
            // Probability scales with food deficit and population (larger = more mouths).
            // Settlements with very low pop (newly founded) are spared.
            let pop_now = world.settlements[i].population;
            let severity = (params.collapse_threshold - food) / params.collapse_threshold.max(0.01);
            let p_death = params.starvation_rate * severity * (pop_now / params.pop_max).min(1.0);
            let roll: f32 = world.rng.random();
            if roll < p_death {
                let x = world.settlements[i].x;
                let y = world.settlements[i].y;
                world.grid[y][x] = TerrainType::Ruin;
                world.settlements[i].alive = false;
            }
        }

        // ── Defense recovery (proportional to population) ────────────────────
        let def = world.settlements[i].defense;
        let pop_now = world.settlements[i].population;
        let def_delta = params.defense_recovery_rate * pop_now;
        world.settlements[i].defense = (def + def_delta).clamp(0.0, 1.0);

        // ── Settlement → Port upgrade ────────────────────────────────────────
        if !world.settlements[i].has_port {
            let x = world.settlements[i].x;
            let y = world.settlements[i].y;

            // Count adjacent ocean tiles (4-connected, cardinal only).
            let n_ocean = world.count_adjacent_ocean(x, y);

            if n_ocean > 0 {
                let prob = params.port_upgrade_prob
                    * world.settlements[i].population
                    * world.settlements[i].food
                    * (1.0 + params.port_ocean_bonus * n_ocean as f32);
                let roll: f32 = world.rng.random();
                if roll < prob {
                    world.settlements[i].has_port = true;
                    world.grid[y][x] = TerrainType::Port;
                }
            }
        }

        // ── Port → Longship building ─────────────────────────────────────────
        if world.settlements[i].has_port && world.settlements[i].wealth >= params.longship_cost {
            let prob = params.longship_build_prob * world.settlements[i].wealth;
            let roll: f32 = world.rng.random();
            if roll < prob {
                world.settlements[i].longships += 1;
                world.settlements[i].wealth -= params.longship_cost;
            }
        }
    }
}
