use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use crate::io::{InitialState, ReplayFrame};
use crate::params::Params;
use crate::settlement::Settlement;
use crate::terrain::TerrainType;

// ── TerrainCounts ─────────────────────────────────────────────────────────────

/// Counts of each terrain type among the 4-connected (cardinal) neighbours of a cell.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TerrainCounts {
    pub plains: u8,
    pub forest: u8,
    pub mountain: u8,
    pub settlement: u8,
    pub ocean: u8,
}

// ── AlivePos ─────────────────────────────────────────────────────────────────

/// Compact (x, y, settlement_index) entry for alive settlements.
/// Packed into 8 bytes for cache-friendly iteration.
#[derive(Clone, Copy)]
pub struct AlivePos {
    pub x: u16,
    pub y: u16,
    pub idx: u32,
}

// ── World ─────────────────────────────────────────────────────────────────────

pub struct World {
    pub width: usize,
    pub height: usize,
    pub grid: Vec<Vec<TerrainType>>,
    pub settlements: Vec<Settlement>,
    pub rng: ChaCha8Rng,
    /// Flat spatial index: `settlement_grid[y * width + x]` → index into
    /// `self.settlements` for the alive settlement at `(x, y)`, or `u32::MAX`
    /// if empty.  Used for O(1) point lookups (`settlement_at`).
    settlement_grid: Vec<u32>,
    /// Compact list of alive settlements: `(x as u16, y as u16, index as u32)`.
    /// Rebuilt when the settlement grid is updated.  Scanned by
    /// `settlements_near` instead of the sparse grid — avoids iterating
    /// empty cells and fits in L1 cache (~200 entries × 8 bytes = 1.6 KB).
    alive_positions: Vec<AlivePos>,
    /// Reusable scratch buffer for shuffled settlement indices.
    pub scratch_indices: Vec<usize>,
    /// Reusable scratch buffer for collecting nearby settlement indices.
    pub scratch_nearby: Vec<usize>,
}

impl World {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build a `World` from a replay frame where all settlement statistics are
    /// known precisely.
    pub fn from_replay_frame(
        frame: &ReplayFrame,
        width: usize,
        height: usize,
        seed: u64,
    ) -> Self {
        let settlements: Vec<Settlement> = frame
            .settlements
            .iter()
            .map(Settlement::from_replay)
            .collect();

        let (settlement_grid, alive_positions) = Self::build_settlement_grid(width, height, &settlements);

        let n = settlements.len();
        World {
            width,
            height,
            grid: frame.grid.clone(),
            settlements,
            rng: ChaCha8Rng::seed_from_u64(seed),
            settlement_grid,
            alive_positions,
            scratch_indices: (0..n).collect(),
            scratch_nearby: Vec::new(),
        }
    }

    /// Build a `World` from an initial state where only structural settlement
    /// data (position, port, alive) is known.  Statistics are sampled from
    /// Normal distributions centred on the `params` priors, with a small
    /// Gaussian noise term.
    pub fn from_initial_state(
        initial: &InitialState,
        params: &Params,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let settlements: Vec<Settlement> = initial
            .settlements
            .iter()
            .enumerate()
            .map(|(i, init)| {
                // Add small fractional noise (~5 %) to each stat.
                let mut noise = |mean: f32| -> f32 {
                    let frac: f32 = rng.random_range(-0.05_f32..=0.05_f32);
                    (mean * (1.0 + frac)).max(0.0)
                };
                Settlement::from_initial(
                    init,
                    noise(params.init_pop_mean),
                    noise(params.init_food_mean),
                    noise(params.init_wealth_mean),
                    noise(params.init_defense_mean),
                    i as u16,
                )
            })
            .collect();

        let width = initial.grid[0].len();
        let height = initial.grid.len();
        let (settlement_grid, alive_positions) = Self::build_settlement_grid(width, height, &settlements);

        let n = settlements.len();
        World {
            width,
            height,
            grid: initial.grid.clone(),
            settlements,
            rng,
            settlement_grid,
            alive_positions,
            scratch_indices: (0..n).collect(),
            scratch_nearby: Vec::new(),
        }
    }

    // ── Grid index helpers ────────────────────────────────────────────────────

    const EMPTY: u32 = u32::MAX;

    /// Build a fresh flat `settlement_grid` and `alive_positions` list.
    fn build_settlement_grid(
        width: usize,
        height: usize,
        settlements: &[Settlement],
    ) -> (Vec<u32>, Vec<AlivePos>) {
        let mut grid = vec![Self::EMPTY; width * height];
        let mut alive = Vec::new();
        for (i, s) in settlements.iter().enumerate() {
            if s.alive {
                grid[s.y * width + s.x] = i as u32;
                alive.push(AlivePos { x: s.x as u16, y: s.y as u16, idx: i as u32 });
            }
        }
        (grid, alive)
    }

    /// Rebuild `self.settlement_grid` to match the current `self.settlements`.
    ///
    /// Call this after any operation that adds or removes settlements, or
    /// changes a settlement's `alive` flag or position.
    pub fn update_settlement_grid(&mut self) {
        self.settlement_grid.fill(Self::EMPTY);
        self.alive_positions.clear();
        let w = self.width;
        for (i, s) in self.settlements.iter().enumerate() {
            if s.alive {
                self.settlement_grid[s.y * w + s.x] = i as u32;
                self.alive_positions.push(AlivePos { x: s.x as u16, y: s.y as u16, idx: i as u32 });
            }
        }
    }

    // ── Spatial queries ───────────────────────────────────────────────────────

    /// Count the 4-connected (cardinal) terrain neighbours of `(x, y)`.
    ///
    /// `x` is the column index, `y` is the row index.  Out-of-bounds
    /// neighbours are skipped silently.
    pub fn count_adjacent_terrain(&self, x: usize, y: usize) -> TerrainCounts {
        let mut counts = TerrainCounts::default();

        let deltas: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in deltas {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            match self.grid[ny][nx] {
                TerrainType::Plains | TerrainType::Empty => counts.plains += 1,
                TerrainType::Forest => counts.forest += 1,
                TerrainType::Mountain => counts.mountain += 1,
                TerrainType::Settlement | TerrainType::Port | TerrainType::Ruin => {
                    counts.settlement += 1
                }
                TerrainType::Ocean => counts.ocean += 1,
            }
        }

        counts
    }

    /// Count 4-connected (cardinal) ocean neighbours of `(x, y)`.
    pub fn count_adjacent_ocean(&self, x: usize, y: usize) -> u32 {
        let mut n = 0;
        if y > 0 && self.grid[y - 1][x].is_ocean() { n += 1; }
        if y + 1 < self.height && self.grid[y + 1][x].is_ocean() { n += 1; }
        if x > 0 && self.grid[y][x - 1].is_ocean() { n += 1; }
        if x + 1 < self.width && self.grid[y][x + 1].is_ocean() { n += 1; }
        n
    }

    /// Return `true` if any of the 4-connected (cardinal) neighbours of
    /// `(x, y)` is `TerrainType::Ocean`.
    pub fn has_adjacent_ocean(&self, x: usize, y: usize) -> bool {
        if y > 0 && self.grid[y - 1][x].is_ocean() {
            return true;
        }
        if y + 1 < self.height && self.grid[y + 1][x].is_ocean() {
            return true;
        }
        if x > 0 && self.grid[y][x - 1].is_ocean() {
            return true;
        }
        if x + 1 < self.width && self.grid[y][x + 1].is_ocean() {
            return true;
        }
        false
    }

    /// Return an iterator over the indices (into `self.settlements`) of all
    /// **alive** settlements whose taxicab (Manhattan) distance from `(x, y)`
    /// is ≤ `max_dist`.
    ///
    /// Dynamically picks the faster strategy:
    /// - **Small radius** (bounding box < n_alive): scan the flat grid
    /// - **Large radius** (bounding box ≥ n_alive): scan the compact alive list
    pub fn settlements_near(
        &self,
        x: usize,
        y: usize,
        max_dist: f32,
    ) -> SettlementsNearIter<'_> {
        let d = max_dist as usize;
        let x_min = x.saturating_sub(d);
        let x_max = (x + d).min(self.width - 1);
        let y_min = y.saturating_sub(d);
        let y_max = (y + d).min(self.height - 1);
        let bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1);

        if bbox_area < self.alive_positions.len() {
            SettlementsNearIter::Grid {
                grid: &self.settlement_grid,
                settlements: &self.settlements,
                cx: x as i32,
                cy: y as i32,
                max_dist_i: max_dist as i32,
                width: self.width,
                x_min,
                x_max,
                y_max,
                nx: x_min,
                ny: y_min,
            }
        } else {
            SettlementsNearIter::List {
                alive: &self.alive_positions,
                settlements: &self.settlements,
                cx: x as i32,
                cy: y as i32,
                max_dist_i: max_dist as i32,
                pos: 0,
            }
        }
    }

    /// Return the index of the **alive** settlement located at exactly `(x, y)`,
    /// or `None` if no such settlement exists.
    ///
    /// O(1) lookup via the settlement grid.  The alive check guards against
    /// stale entries that may exist when a settlement was killed mid-step
    /// (conflict / winter) but the grid hasn't been rebuilt yet.
    pub fn settlement_at(&self, x: usize, y: usize) -> Option<usize> {
        let idx = self.settlement_grid[y * self.width + x];
        if idx != Self::EMPTY && self.settlements[idx as usize].alive {
            Some(idx as usize)
        } else {
            None
        }
    }

    // ── Snapshot ──────────────────────────────────────────────────────────────

    /// Produce a grid of prediction class indices (0–5) corresponding to the
    /// current terrain.
    pub fn prediction_snapshot(&self) -> Vec<Vec<usize>> {
        self.grid
            .iter()
            .map(|row| row.iter().map(|t| t.prediction_class()).collect())
            .collect()
    }
}

// ── SettlementsNearIter ───────────────────────────────────────────────────────

/// Zero-allocation iterator over settlement indices near a point.
/// Dynamically uses grid scan (small radius) or alive-list scan (large radius).
pub enum SettlementsNearIter<'a> {
    /// Scan the flat settlement grid within a bounding box.
    Grid {
        grid: &'a [u32],
        settlements: &'a [Settlement],
        cx: i32,
        cy: i32,
        max_dist_i: i32,
        width: usize,
        x_min: usize,
        x_max: usize,
        y_max: usize,
        nx: usize,
        ny: usize,
    },
    /// Scan the compact alive-positions list.
    List {
        alive: &'a [AlivePos],
        settlements: &'a [Settlement],
        cx: i32,
        cy: i32,
        max_dist_i: i32,
        pos: usize,
    },
}

impl<'a> Iterator for SettlementsNearIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        match self {
            SettlementsNearIter::Grid {
                grid, settlements, cx, cy, max_dist_i,
                width, x_min, x_max, y_max, nx, ny,
            } => {
                loop {
                    if *ny > *y_max {
                        return None;
                    }
                    let cur_x = *nx;
                    let cur_y = *ny;

                    *nx += 1;
                    if *nx > *x_max {
                        *nx = *x_min;
                        *ny += 1;
                    }

                    let idx = grid[cur_y * *width + cur_x];
                    if idx != World::EMPTY {
                        let idx = idx as usize;
                        if !settlements[idx].alive {
                            continue;
                        }
                        let dx = (cur_x as i32 - *cx).abs();
                        let dy = (cur_y as i32 - *cy).abs();
                        if dx + dy <= *max_dist_i {
                            return Some(idx);
                        }
                    }
                }
            }
            SettlementsNearIter::List {
                alive, settlements, cx, cy, max_dist_i, pos,
            } => {
                while *pos < alive.len() {
                    let ap = alive[*pos];
                    *pos += 1;

                    let idx = ap.idx as usize;
                    if !settlements[idx].alive {
                        continue;
                    }

                    let dx = (ap.x as i32 - *cx).abs();
                    let dy = (ap.y as i32 - *cy).abs();
                    if dx + dy <= *max_dist_i {
                        return Some(idx);
                    }
                }
                None
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ReplayData;

    const ROUND_ID: &str = "ae78003a-4efe-425a-881a-d16a39bca0ad";

    #[test]
    fn test_world_from_replay_frame_dimensions() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        assert_eq!(world.width, replay.width);
        assert_eq!(world.height, replay.height);
        assert_eq!(world.grid.len(), replay.height);
        assert_eq!(world.grid[0].len(), replay.width);
        assert_eq!(world.settlements.len(), frame.settlements.len());
    }

    #[test]
    fn test_world_from_replay_frame_settlements_alive() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        // Count alive settlements matches between frame and world.
        let frame_alive = frame.settlements.iter().filter(|s| s.alive).count();
        let world_alive = world.settlements.iter().filter(|s| s.alive).count();
        assert_eq!(frame_alive, world_alive);
    }

    #[test]
    fn test_count_adjacent_terrain_corner() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        // Corner (0,0) has at most 2 neighbours — just verify it doesn't panic.
        let _counts = world.count_adjacent_terrain(0, 0);
    }

    #[test]
    fn test_count_adjacent_terrain_interior_sums() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        // An interior cell must have exactly 4 neighbours in total.
        let x = world.width / 2;
        let y = world.height / 2;
        let c = world.count_adjacent_terrain(x, y);
        let total = c.plains as u32
            + c.forest as u32
            + c.mountain as u32
            + c.settlement as u32
            + c.ocean as u32;
        assert_eq!(total, 4);
    }

    #[test]
    fn test_settlements_near_returns_alive_only() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        // With a very large radius every alive settlement should be returned.
        let large_radius = (world.width.max(world.height) * 2) as f32;
        let near: Vec<usize> = world.settlements_near(world.width / 2, world.height / 2, large_radius).collect();
        let alive_count = world.settlements.iter().filter(|s| s.alive).count();
        assert_eq!(near.len(), alive_count);
    }

    #[test]
    fn test_settlement_at_finds_existing() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        // Pick the first alive settlement and look it up.
        if let Some(s) = world.settlements.iter().find(|s| s.alive) {
            let idx = world.settlement_at(s.x, s.y);
            assert!(idx.is_some(), "expected to find settlement at ({}, {})", s.x, s.y);
        }
    }

    #[test]
    fn test_prediction_snapshot_shape() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        let frame = &replay.frames[0];
        let world =
            World::from_replay_frame(frame, replay.width, replay.height, replay.sim_seed);

        let snap = world.prediction_snapshot();
        assert_eq!(snap.len(), world.height);
        assert_eq!(snap[0].len(), world.width);
        // All values must be 0–5.
        for row in &snap {
            for &v in row {
                assert!(v <= 5, "unexpected prediction class {}", v);
            }
        }
    }

    #[test]
    fn test_world_from_initial_state() {
        use crate::io::RoundDetails;

        let details = RoundDetails::load(ROUND_ID).expect("load details");
        let params = Params::default_prior();
        let initial = &details.initial_states[0];
        let world = World::from_initial_state(initial, &params, 42);

        assert_eq!(world.width, details.map_width);
        assert_eq!(world.height, details.map_height);
        assert_eq!(world.settlements.len(), initial.settlements.len());
        // Stats should be positive.
        for s in &world.settlements {
            assert!(s.population >= 0.0);
            assert!(s.food >= 0.0);
        }
    }
}
