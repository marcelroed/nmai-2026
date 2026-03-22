use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::params::Params;
use crate::settlement::Settlement;
use crate::terrain::TerrainType;
use crate::world::World;

/// Count adjacent food-producing tiles (Plains and Forest) in 4-connected neighborhood.
fn count_adjacent_food_tiles(world: &World, x: usize, y: usize) -> u32 {
    let counts = world.count_adjacent_terrain(x, y);
    counts.plains as u32 + counts.forest as u32
}

/// Pick a spawn cell using one-pass weighted reservoir sampling.
/// Returns `Some((x, y))` for the chosen cell, or `None` if no valid candidate exists.
/// Avoids allocating a candidates Vec entirely.
fn pick_spawn_cell(
    world: &World,
    ox: usize,
    oy: usize,
    max_dist: f32,
    params: &Params,
    new_settlements: &[Settlement],
    rng: &mut impl rand::Rng,
) -> Option<(usize, usize)> {
    let max_d = max_dist as i32;
    let ox_i = ox as i32;
    let oy_i = oy as i32;

    let x_min = (ox_i - max_d).max(0) as usize;
    let x_max = ((ox_i + max_d) as usize).min(world.width - 1);
    let y_min = (oy_i - max_d).max(0) as usize;
    let y_max = ((oy_i + max_d) as usize).min(world.height - 1);

    let mut chosen: Option<(usize, usize)> = None;
    let mut total_weight: f32 = 0.0;

    for cy in y_min..=y_max {
        for cx in x_min..=x_max {
            if cx == ox && cy == oy {
                continue;
            }
            let cdx = (cx as i32 - ox_i).unsigned_abs() as f32;
            let cdy = (cy as i32 - oy_i).unsigned_abs() as f32;
            let taxi_dist = cdx + cdy;
            if taxi_dist > max_dist {
                continue;
            }

            let terrain = world.grid[cy][cx];
            if !matches!(terrain, TerrainType::Plains | TerrainType::Forest) {
                continue;
            }

            if world.settlement_at(cx, cy).is_some() {
                continue;
            }

            // Skip cells claimed by newly-spawned settlements this step.
            if new_settlements.iter().any(|ns| ns.x == cx && ns.y == cy) {
                continue;
            }

            let dist_weight = (-params.spawn_distance_decay * taxi_dist).exp();
            let terrain_weight = match terrain {
                TerrainType::Forest => params.spawn_weight_forest,
                _ => 1.0,
            };
            let adj_food = count_adjacent_food_tiles(world, cx, cy);
            let terrain_bonus = 1.0 + params.spawn_terrain_bonus * adj_food as f32;
            let weight = (dist_weight * terrain_weight * terrain_bonus).max(0.001);

            total_weight += weight;
            if rng.random::<f32>() < weight / total_weight {
                chosen = Some((cx, cy));
            }
        }
    }

    chosen
}

/// Phase 3: Settlement Expansion.
///
/// Each alive settlement may probabilistically spawn a child settlement on a
/// nearby buildable cell.  New settlements are collected and appended to
/// `world.settlements` after all parents have been processed.
pub fn phase_growth(world: &mut World, params: &Params) {
    // Shuffle the indices so spawn order is random (reuse scratch buffer).
    let n = world.settlements.len();
    let mut indices = std::mem::take(&mut world.scratch_indices);
    indices.resize(n, 0);
    for i in 0..n { indices[i] = i; }
    indices.shuffle(&mut world.rng);

    let mut new_settlements: Vec<Settlement> = Vec::new();

    for &idx in &indices {
        if !world.settlements[idx].alive {
            continue;
        }

        let (ox, oy, pop, food, owner_id, tech) = {
            let s = &world.settlements[idx];
            (s.x, s.y, s.population, s.food, s.owner_id, s.tech_level)
        };

        // Spawn probability: includes tech boost.
        let p_spawn = params.growth_prob * pop * food
            * (1.0 + params.tech_growth_coeff * tech);
        let roll: f32 = world.rng.random();
        if roll >= p_spawn {
            continue;
        }

        // One-pass weighted reservoir sampling: pick a spawn cell without
        // allocating a candidates Vec.
        let mut rng = std::mem::replace(&mut world.rng, ChaCha8Rng::seed_from_u64(0));
        let picked = pick_spawn_cell(
            world, ox, oy, params.spawn_distance_max, params,
            &new_settlements, &mut rng,
        );
        world.rng = rng;
        let (tx, ty) = match picked {
            Some(pos) => pos,
            None => continue,
        };

        let child_pop = params.child_pop;
        let child_def = 0.2_f32;

        if pop <= params.parent_cost {
            continue;
        }

        world.settlements[idx].population -= params.parent_cost;

        let is_port = false;

        let child_food = world.settlements[idx].food * 0.2;
        let child_wealth = world.settlements[idx].wealth * 0.2;

        let new_s = Settlement {
            x: tx,
            y: ty,
            population: child_pop,
            food: child_food,
            wealth: child_wealth,
            defense: child_def,
            has_port: is_port,
            alive: true,
            owner_id,
            tech_level: 0.0,
            longships: 0,
            total_damage: 0.0,
        };

        // Update grid.
        world.grid[ty][tx] = if is_port {
            TerrainType::Port
        } else {
            TerrainType::Settlement
        };

        new_settlements.push(new_s);
    }

    world.scratch_indices = indices;
    world.settlements.extend(new_settlements);

    world.update_settlement_grid();
}
