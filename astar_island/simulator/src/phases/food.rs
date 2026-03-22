use crate::params::Params;
use crate::world::World;

/// Phase 1: Food Production.
///
/// Nonlinear logistic food model (proven from replay data):
///
///   Δfood = food_plains   × n_plains     × (1 − food)
///         + food_forest   × n_forest     × (1 − food)
///         + food_mountain × n_mountain   × (1 − food)
///         + food_settlement × n_settlement × (1 − food)
///         + food_feedback × food × (1 − food)
///         + food_pop_coeff × population
///
/// The (1 − food) factor creates logistic saturation: production slows as
/// food approaches capacity.  Population drains food linearly.
///
/// Food is clamped to [0, 0.998].
pub fn phase_food(world: &mut World, params: &Params) {
    for s in world.settlements.iter_mut() {
        if !s.alive {
            continue;
        }

        let counts = {
            let x = s.x as i32;
            let y = s.y as i32;
            let width = world.width as i32;
            let height = world.height as i32;
            let grid = &world.grid;

            let mut n_plains = 0u32;
            let mut n_forest = 0u32;
            let mut n_mountain = 0u32;
            let mut n_settlement = 0u32;
            let mut n_ocean = 0u32;

            let deltas: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for (dx, dy) in deltas {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= width || ny >= height {
                    continue;
                }
                use crate::terrain::TerrainType;
                match grid[ny as usize][nx as usize] {
                    TerrainType::Plains | TerrainType::Empty => n_plains += 1,
                    TerrainType::Forest => n_forest += 1,
                    TerrainType::Mountain => n_mountain += 1,
                    TerrainType::Settlement
                    | TerrainType::Port
                    | TerrainType::Ruin => n_settlement += 1,
                    TerrainType::Ocean => n_ocean += 1,
                }
            }
            (n_plains, n_forest, n_mountain, n_settlement, n_ocean)
        };

        let (n_plains, n_forest, n_mountain, n_settlement, n_ocean) = counts;
        let omf = 1.0 - s.food;

        let food_delta = params.food_plains * n_plains as f32 * omf
            + params.food_forest * n_forest as f32 * omf
            + params.food_mountain * n_mountain as f32 * omf
            + params.food_settlement * n_settlement as f32 * omf
            + params.food_ocean * n_ocean as f32 * omf
            + params.food_feedback * s.food * omf
            + params.food_pop_coeff * s.population
            + params.tech_food_coeff * s.tech_level * omf;

        s.food = (s.food + food_delta).clamp(0.0, 0.998);
    }
}
