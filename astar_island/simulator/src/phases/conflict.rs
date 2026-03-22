use rand::prelude::*;

use crate::params::Params;
use crate::terrain::TerrainType;
use crate::world::World;

/// Phase 4: Raiding.
///
/// Each alive settlement may raid nearby settlements belonging to a different
/// faction.  Raid probability decays with taxicab distance and increases with
/// hunger.
///
/// On each raid:
///   1. Defender defense is multiplied by `raid_defense_mult`.
///   2. With probability `raid_success_prob`, wealth transfers.
///   3. Damage accumulates: `total_damage += raid_damage_inc`.
///   4. Probabilistic outcome based on accumulated damage:
///      - p_ruin = total_damage * raid_kill_scale → settlement becomes ruin.
///      - p_takeover = total_damage * raid_takeover_scale → change owner (damage resets).
///      Ruin is checked first.
pub fn phase_conflict(world: &mut World, params: &Params) {
    let n = world.settlements.len();
    let mut indices = std::mem::take(&mut world.scratch_indices);
    indices.resize(n, 0);
    for i in 0..n { indices[i] = i; }
    indices.shuffle(&mut world.rng);

    let mut targets = std::mem::take(&mut world.scratch_nearby);

    for &att_idx in &indices {
        if !world.settlements[att_idx].alive {
            continue;
        }

        let (att_x, att_y, att_food, att_longships, att_owner) = {
            let a = &world.settlements[att_idx];
            (a.x, a.y, a.food, a.longships, a.owner_id)
        };

        // Range depends on longships.
        let range = params.raid_range
            + if att_longships > 0 { params.longship_range_bonus } else { 0.0 };

        // Collect potential target indices into reusable buffer.
        targets.clear();
        targets.extend(
            world.settlements_near(att_x, att_y, range)
                .filter(|&i| i != att_idx && world.settlements[i].owner_id != att_owner)
        );

        for &def_idx in &targets {
            if !world.settlements[def_idx].alive {
                continue;
            }

            // Raid probability: scales with hunger and decays with distance.
            let def = &world.settlements[def_idx];
            let taxi_dist = (def.x as f32 - att_x as f32).abs()
                + (def.y as f32 - att_y as f32).abs();
            let hunger_factor = (1.0 - att_food).max(0.0);
            let distance_factor = (-params.raid_distance_decay * taxi_dist).exp();
            let p_raid =
                params.raid_prob * (1.0 + params.aggression * hunger_factor) * distance_factor;
            let roll: f32 = world.rng.random();
            if roll >= p_raid {
                continue;
            }

            // Raid happens.

            // 1. Defense reduction (multiplicative).
            world.settlements[def_idx].defense *= params.raid_defense_mult;

            // 2. Wealth theft (with success probability).
            if world.rng.random::<f32>() < params.raid_success_prob {
                let stolen = params.raid_steal_frac * world.settlements[def_idx].wealth;
                world.settlements[def_idx].wealth -= stolen;
                world.settlements[att_idx].wealth += stolen;
            }

            // 3. Accumulate damage.
            world.settlements[def_idx].total_damage += params.raid_damage_inc;
            let dmg = world.settlements[def_idx].total_damage;

            // 4. Probabilistic outcome scaled by accumulated damage.
            let p_ruin = (dmg * params.raid_kill_scale).min(1.0);
            let p_takeover = (dmg * params.raid_takeover_scale).min(1.0);

            let roll: f32 = world.rng.random();
            if roll < p_ruin {
                // Destruction.
                let dx = world.settlements[def_idx].x;
                let dy = world.settlements[def_idx].y;
                world.grid[dy][dx] = TerrainType::Ruin;
                world.settlements[def_idx].alive = false;
            } else if roll < p_ruin + p_takeover {
                // Takeover: change owner, reset damage.
                world.settlements[def_idx].owner_id = att_owner;
                world.settlements[def_idx].total_damage = 0.0;
            }
        }
    }
    world.scratch_indices = indices;
    world.scratch_nearby = targets;
}
