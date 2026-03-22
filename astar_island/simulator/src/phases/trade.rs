use crate::params::Params;
use crate::world::World;

/// Phase 5: Port-to-Port Trade.
///
/// For each pair of alive port settlements sharing the same owner within
/// `trade_range` (taxicab distance), both ports gain food and wealth.
/// Tech levels also diffuse between connected ports, moving toward their
/// average at `tech_diffusion_rate`.
pub fn phase_trade(world: &mut World, params: &Params) {
    let n = world.settlements.len();

    // Collect (index, x, y, owner_id) for alive ports to avoid repeated
    // iteration.
    let ports: Vec<(usize, f32, f32, u16)> = world
        .settlements
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            if s.alive && s.has_port {
                Some((i, s.x as f32, s.y as f32, s.owner_id))
            } else {
                None
            }
        })
        .collect();

    // Accumulate bonuses and tech deltas per settlement.
    let mut food_bonus = vec![0.0_f32; n];
    let mut wealth_bonus = vec![0.0_f32; n];
    let mut tech_delta = vec![0.0_f32; n];

    for i in 0..ports.len() {
        for j in (i + 1)..ports.len() {
            let (ia, xa, ya, oa) = ports[i];
            let (ib, xb, yb, ob) = ports[j];

            if oa != ob {
                continue;
            }
            let dx = (xa - xb).abs();
            let dy = (ya - yb).abs();
            if dx + dy > params.trade_range {
                continue;
            }

            food_bonus[ia] += params.trade_food_bonus;
            food_bonus[ib] += params.trade_food_bonus;
            wealth_bonus[ia] += params.trade_wealth_bonus;
            wealth_bonus[ib] += params.trade_wealth_bonus;

            // Tech diffusion: the lower-tech port gains toward the higher.
            // No port loses tech from trading.
            let tech_a = world.settlements[ia].tech_level;
            let tech_b = world.settlements[ib].tech_level;
            if tech_b > tech_a {
                tech_delta[ia] += params.tech_diffusion_rate * (tech_b - tech_a);
            } else if tech_a > tech_b {
                tech_delta[ib] += params.tech_diffusion_rate * (tech_a - tech_b);
            }
        }
    }

    // Apply accumulated bonuses, tech diffusion, and wealth decay.
    for (i, s) in world.settlements.iter_mut().enumerate() {
        if !s.alive {
            continue;
        }
        if food_bonus[i] != 0.0 || wealth_bonus[i] != 0.0 {
            s.food = (s.food + food_bonus[i]).min(0.998);
            s.wealth += wealth_bonus[i];
        }
        if tech_delta[i] != 0.0 {
            s.tech_level = (s.tech_level + tech_delta[i]).clamp(0.0, 1.0);
        }
        // Proportional wealth decay (~1.4% per step from replay data).
        s.wealth *= 1.0 - params.wealth_decay;
    }
}
