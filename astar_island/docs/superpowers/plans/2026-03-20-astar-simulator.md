# Astar Island Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust simulator that replicates the Astar Island Norse civilisation dynamics, infer hidden parameters from viewport queries via CMA-ES, and run Monte Carlo to produce maximum-score terrain probability predictions.

**Architecture:** New Rust crate (`simulator/`) separate from the existing `astar-rs/` visualizer (which has TUI dependencies we don't need). The simulator crate is self-contained with its own data loading — `astar-rs/` remains untouched for visualization. Python orchestrator calls Rust via CLI for API interaction, query planning, and submission. Validation against 45 replays from 9 historical rounds.

**Tech Stack:** Rust (rayon, rand, serde_json), Python (requests, numpy), CMA-ES for parameter optimization.

**Spec:** `docs/superpowers/specs/2026-03-20-astar-island-simulator-design.md`

---

## File Structure

### New Rust Crate: `simulator/`

| File | Responsibility |
|------|---------------|
| `simulator/Cargo.toml` | Crate config with dependencies |
| `simulator/src/lib.rs` | Public API re-exports |
| `simulator/src/terrain.rs` | TerrainType enum, prediction class mapping |
| `simulator/src/settlement.rs` | Settlement struct and helpers |
| `simulator/src/params.rs` | Params struct with defaults and bounds |
| `simulator/src/world.rs` | World state: grid + settlements + spatial index |
| `simulator/src/io.rs` | JSON loading: replays, ground truth, details, queries |
| `simulator/src/phases/mod.rs` | Phase trait + step orchestration |
| `simulator/src/phases/food.rs` | Phase 1: food production |
| `simulator/src/phases/pop_defense.rs` | Phase 2: population & defense dynamics |
| `simulator/src/phases/growth.rs` | Phase 3: settlement expansion |
| `simulator/src/phases/conflict.rs` | Phase 4: raiding + takeover |
| `simulator/src/phases/trade.rs` | Phase 5: port-to-port trade |
| `simulator/src/phases/winter.rs` | Phase 6: winter food loss + collapse |
| `simulator/src/phases/environment.rs` | Phase 7: ruin transitions |
| `simulator/src/montecarlo.rs` | Run N sims, aggregate to probability distribution |
| `simulator/src/inference.rs` | CMA-ES parameter search |
| `simulator/src/scoring.rs` | Entropy-weighted KL divergence scoring |
| `simulator/src/main.rs` | CLI entry points: simulate, montecarlo, infer, score |

### Modified Python Files: `src/astar_island/`

| File | Responsibility |
|------|---------------|
| `src/astar_island/orchestrator.py` | End-to-end pipeline: fetch, infer, predict, submit |
| `src/astar_island/query_planner.py` | Strategic viewport placement |

---

## Task 1: Rust Crate Scaffold + Core Types

**Files:**
- Create: `simulator/Cargo.toml`
- Create: `simulator/src/lib.rs`
- Create: `simulator/src/terrain.rs`
- Create: `simulator/src/settlement.rs`
- Create: `simulator/src/params.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "astar-simulator"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = "1.0"
rand = "0.9"
rand_chacha = "0.9"
rayon = "1.10"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

- [ ] **Step 2: Create terrain.rs**

```rust
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TerrainType {
    Empty = 0,
    Settlement = 1,
    Port = 2,
    Ruin = 3,
    Forest = 4,
    Mountain = 5,
    Ocean = 10,
    Plains = 11,
}

impl TerrainType {
    pub fn from_code(code: i64) -> Option<Self> {
        match code {
            0 => Some(Self::Empty),
            1 => Some(Self::Settlement),
            2 => Some(Self::Port),
            3 => Some(Self::Ruin),
            4 => Some(Self::Forest),
            5 => Some(Self::Mountain),
            10 => Some(Self::Ocean),
            11 => Some(Self::Plains),
            _ => None,
        }
    }

    /// Map internal terrain to one of 6 prediction classes.
    pub fn prediction_class(self) -> usize {
        match self {
            Self::Empty | Self::Ocean | Self::Plains => 0,
            Self::Settlement => 1,
            Self::Port => 2,
            Self::Ruin => 3,
            Self::Forest => 4,
            Self::Mountain => 5,
        }
    }

    /// Can a new settlement be built on this terrain?
    pub fn is_buildable(self) -> bool {
        matches!(self, Self::Plains | Self::Ruin | Self::Forest)
    }

    /// Is this terrain static (never changes)?
    pub fn is_static(self) -> bool {
        matches!(self, Self::Ocean | Self::Mountain)
    }

    /// Is this an ocean tile?
    pub fn is_ocean(self) -> bool {
        matches!(self, Self::Ocean)
    }
}

// Custom deserializer using Visitor to handle both i64 and u64 JSON integers
impl<'de> Deserialize<'de> for TerrainType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct TerrainVisitor;
        impl serde::de::Visitor<'_> for TerrainVisitor {
            type Value = TerrainType;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a terrain code integer")
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                TerrainType::from_code(v).ok_or_else(|| E::custom(format!("unknown terrain: {v}")))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                self.visit_i64(v as i64)
            }
        }
        deserializer.deserialize_i64(TerrainVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_classes() {
        assert_eq!(TerrainType::Ocean.prediction_class(), 0);
        assert_eq!(TerrainType::Plains.prediction_class(), 0);
        assert_eq!(TerrainType::Empty.prediction_class(), 0);
        assert_eq!(TerrainType::Settlement.prediction_class(), 1);
        assert_eq!(TerrainType::Port.prediction_class(), 2);
        assert_eq!(TerrainType::Ruin.prediction_class(), 3);
        assert_eq!(TerrainType::Forest.prediction_class(), 4);
        assert_eq!(TerrainType::Mountain.prediction_class(), 5);
    }

    #[test]
    fn test_buildable() {
        assert!(TerrainType::Plains.is_buildable());
        assert!(TerrainType::Ruin.is_buildable());
        assert!(TerrainType::Forest.is_buildable());
        assert!(!TerrainType::Ocean.is_buildable());
        assert!(!TerrainType::Mountain.is_buildable());
        assert!(!TerrainType::Settlement.is_buildable());
    }
}
```

- [ ] **Step 3: Create settlement.rs**

```rust
use serde::Deserialize;

#[derive(Clone, Debug)]
pub struct Settlement {
    pub x: usize,
    pub y: usize,
    pub population: f32,
    pub food: f32,
    pub wealth: f32,
    pub defense: f32,
    pub has_port: bool,
    pub alive: bool,
    pub owner_id: u16,
}

/// Settlement as it appears in replay JSON (with all stats).
#[derive(Clone, Debug, Deserialize)]
pub struct ReplaySettlement {
    pub x: usize,
    pub y: usize,
    pub population: f64,
    pub food: f64,
    pub wealth: f64,
    pub defense: f64,
    pub has_port: bool,
    pub alive: bool,
    pub owner_id: u16,
}

/// Settlement as it appears in initial_states JSON (stats hidden).
#[derive(Clone, Debug, Deserialize)]
pub struct InitialSettlement {
    pub x: usize,
    pub y: usize,
    pub has_port: bool,
    pub alive: bool,
}

impl Settlement {
    /// Create from replay data (full stats known).
    pub fn from_replay(rs: &ReplaySettlement) -> Self {
        Self {
            x: rs.x,
            y: rs.y,
            population: rs.population as f32,
            food: rs.food as f32,
            wealth: rs.wealth as f32,
            defense: rs.defense as f32,
            has_port: rs.has_port,
            alive: rs.alive,
            owner_id: rs.owner_id,
        }
    }

    /// Create from initial state (stats inferred from params).
    pub fn from_initial(is: &InitialSettlement, pop: f32, food: f32, wealth: f32, defense: f32, owner_id: u16) -> Self {
        Self {
            x: is.x,
            y: is.y,
            population: pop,
            food,
            wealth,
            defense,
            has_port: is.has_port,
            alive: is.alive,
            owner_id,
        }
    }
}
```

- [ ] **Step 4: Create params.rs**

```rust
/// Hidden simulation parameters. Values vary per round.
#[derive(Clone, Debug)]
pub struct Params {
    // Food production (Phase 1)
    pub food_base: f32,
    pub food_pop_coeff: f32,
    pub food_feedback: f32,
    pub food_plains: f32,
    pub food_forest: f32,
    pub food_mountain: f32,
    pub food_settlement: f32,

    // Population dynamics (Phase 2)
    pub pop_growth_rate: f32,
    pub pop_max: f32,

    // Defense dynamics (Phase 2)
    pub defense_recovery_rate: f32,

    // Growth / expansion (Phase 3)
    pub growth_prob: f32,
    pub spawn_distance_max: f32,

    // Conflict (Phase 4)
    pub raid_prob: f32,
    pub raid_range: f32,
    pub longship_range_bonus: f32,
    pub aggression: f32,
    pub takeover_threshold: f32,
    pub raid_damage: f32,

    // Trade (Phase 5)
    pub trade_range: f32,
    pub trade_food_bonus: f32,
    pub trade_wealth_bonus: f32,

    // Winter (Phase 6)
    pub winter_severity: f32,
    pub collapse_threshold: f32,

    // Environment (Phase 7)
    pub ruin_to_settlement: f32,
    pub ruin_to_forest: f32,
    pub reclaim_range: f32,

    // Initial settlement stats (inferred)
    pub init_pop_mean: f32,
    pub init_food_mean: f32,
    pub init_wealth_mean: f32,
    pub init_defense_mean: f32,
}

impl Params {
    /// Default params from cross-round averages (useful as CMA-ES starting point).
    pub fn default_prior() -> Self {
        Self {
            food_base: 0.43,
            food_pop_coeff: -0.11,
            food_feedback: -0.45,
            food_plains: 0.011,
            food_forest: 0.017,
            food_mountain: -0.003,
            food_settlement: -0.007,
            pop_growth_rate: 0.05,
            pop_max: 5.0,
            defense_recovery_rate: 0.02,
            growth_prob: 0.15,
            spawn_distance_max: 3.0,
            raid_prob: 0.1,
            raid_range: 3.0,
            longship_range_bonus: 2.0,
            aggression: 1.0,
            takeover_threshold: 0.15,
            raid_damage: 0.1,
            trade_range: 5.0,
            trade_food_bonus: 0.02,
            trade_wealth_bonus: 0.01,
            winter_severity: 0.1,
            collapse_threshold: 0.05,
            ruin_to_settlement: 0.47,
            ruin_to_forest: 0.17,
            reclaim_range: 3.0,
            init_pop_mean: 1.0,
            init_food_mean: 0.55,
            init_wealth_mean: 0.30,
            init_defense_mean: 0.40,
        }
    }

    /// Pack params into a flat f32 vector (for CMA-ES).
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.food_base, self.food_pop_coeff, self.food_feedback,
            self.food_plains, self.food_forest, self.food_mountain, self.food_settlement,
            self.pop_growth_rate, self.pop_max,
            self.defense_recovery_rate,
            self.growth_prob, self.spawn_distance_max,
            self.raid_prob, self.raid_range, self.longship_range_bonus,
            self.aggression, self.takeover_threshold, self.raid_damage,
            self.trade_range, self.trade_food_bonus, self.trade_wealth_bonus,
            self.winter_severity, self.collapse_threshold,
            self.ruin_to_settlement, self.ruin_to_forest, self.reclaim_range,
            self.init_pop_mean, self.init_food_mean, self.init_wealth_mean, self.init_defense_mean,
        ]
    }

    /// Unpack params from a flat f32 vector.
    pub fn from_vec(v: &[f32]) -> Self {
        assert_eq!(v.len(), 30, "expected 30 params, got {}", v.len());
        Self {
            food_base: v[0], food_pop_coeff: v[1], food_feedback: v[2],
            food_plains: v[3], food_forest: v[4], food_mountain: v[5], food_settlement: v[6],
            pop_growth_rate: v[7], pop_max: v[8],
            defense_recovery_rate: v[9],
            growth_prob: v[10], spawn_distance_max: v[11],
            raid_prob: v[12], raid_range: v[13], longship_range_bonus: v[14],
            aggression: v[15], takeover_threshold: v[16], raid_damage: v[17],
            trade_range: v[18], trade_food_bonus: v[19], trade_wealth_bonus: v[20],
            winter_severity: v[21], collapse_threshold: v[22],
            ruin_to_settlement: v[23], ruin_to_forest: v[24], reclaim_range: v[25],
            init_pop_mean: v[26], init_food_mean: v[27], init_wealth_mean: v[28], init_defense_mean: v[29],
        }
    }

    /// Lower bounds for CMA-ES search.
    pub fn lower_bounds() -> Vec<f32> {
        vec![
            0.20, -0.25, -0.70, 0.0, 0.0, -0.02, -0.02,  // food
            0.0, 2.0,                                        // pop
            0.0,                                              // defense
            0.01, 1.5,                                        // growth
            0.01, 1.5, 0.0, 0.0, 0.05, 0.02,                // conflict
            2.0, 0.0, 0.0,                                    // trade
            0.01, 0.0,                                         // winter
            0.20, 0.05, 1.0,                                   // environment
            0.5, 0.2, 0.05, 0.1,                               // init stats
        ]
    }

    /// Upper bounds for CMA-ES search.
    pub fn upper_bounds() -> Vec<f32> {
        vec![
            0.70, 0.0, -0.20, 0.04, 0.04, 0.01, 0.01,      // food
            0.20, 8.0,                                        // pop
            0.10,                                              // defense
            0.50, 5.0,                                         // growth
            0.40, 6.0, 5.0, 3.0, 0.50, 0.30,                 // conflict
            10.0, 0.10, 0.05,                                  // trade
            0.40, 0.20,                                        // winter
            0.65, 0.30, 5.0,                                   // environment
            1.5, 0.9, 0.6, 0.7,                                // init stats
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_vec() {
        let p = Params::default_prior();
        let v = p.to_vec();
        assert_eq!(v.len(), 30);
        let p2 = Params::from_vec(&v);
        assert_eq!(p2.to_vec(), v);
    }

    #[test]
    fn test_bounds_length() {
        assert_eq!(Params::lower_bounds().len(), 30);
        assert_eq!(Params::upper_bounds().len(), 30);
    }
}
```

- [ ] **Step 5: Create lib.rs**

```rust
pub mod terrain;
pub mod settlement;
pub mod params;

pub use terrain::TerrainType;
pub use settlement::Settlement;
pub use params::Params;
```

- [ ] **Step 6: Build and run tests**

Run: `cd simulator && cargo test`
Expected: All tests pass, crate compiles.

- [ ] **Step 7: Commit**

```bash
git add simulator/
git commit -m "feat: scaffold simulator crate with core types (terrain, settlement, params)"
```

---

## Task 2: World State + Spatial Index + IO

**Files:**
- Create: `simulator/src/world.rs`
- Create: `simulator/src/io.rs`
- Modify: `simulator/src/lib.rs`

- [ ] **Step 1: Create io.rs — JSON data loading**

Load replay, ground truth, and details files from the `data/` directory. Structures mirror the JSON format exactly.

```rust
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use serde::Deserialize;

use crate::terrain::TerrainType;
use crate::settlement::{ReplaySettlement, InitialSettlement};

pub type Grid = Vec<Vec<TerrainType>>;
pub type DistributionGrid = Vec<Vec<[f64; 6]>>;

pub fn data_dir() -> PathBuf {
    // Check env var first, then fall back to relative paths
    if let Ok(dir) = std::env::var("ASTAR_DATA_DIR") {
        return PathBuf::from(dir);
    }
    for candidate in ["../data", "data", "astar_island/data"] {
        let p = PathBuf::from(candidate);
        if p.is_dir() { return p; }
    }
    panic!("Could not find data directory. Set ASTAR_DATA_DIR env var or run from astar_island/ or simulator/.");
}

#[derive(Debug, Deserialize)]
pub struct ReplayFrame {
    pub step: u32,
    pub grid: Grid,
    pub settlements: Vec<ReplaySettlement>,
}

#[derive(Debug, Deserialize)]
pub struct ReplayData {
    pub round_id: String,
    pub seed_index: usize,
    pub sim_seed: u64,
    pub width: usize,
    pub height: usize,
    pub frames: Vec<ReplayFrame>,
}

#[derive(Debug, Deserialize)]
pub struct GroundTruthData {
    pub width: usize,
    pub height: usize,
    pub initial_grid: Grid,
    pub ground_truth: DistributionGrid,
    pub prediction: Option<DistributionGrid>,
    pub score: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct InitialState {
    pub grid: Grid,
    pub settlements: Vec<InitialSettlement>,
}

#[derive(Debug, Deserialize)]
pub struct RoundDetails {
    pub id: String,
    pub round_number: u32,
    pub map_width: usize,
    pub map_height: usize,
    pub seeds_count: usize,
    pub initial_states: Vec<InitialState>,
}

#[derive(Debug, Deserialize)]
pub struct QueryViewport {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

#[derive(Debug, Deserialize)]
pub struct QueryResult {
    pub grid: Grid,
    pub settlements: Vec<ReplaySettlement>,
    pub viewport: QueryViewport,
    pub width: usize,
    pub height: usize,
    pub queries_used: u32,
    pub queries_max: u32,
}

impl ReplayData {
    pub fn load(round_id: &str, seed_index: usize) -> Result<Self> {
        let path = data_dir()
            .join(round_id)
            .join("analysis")
            .join(format!("replay_seed_index={seed_index}.json"));
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading replay {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing replay {}", path.display()))
    }
}

impl GroundTruthData {
    pub fn load(round_id: &str, seed_index: usize) -> Result<Self> {
        let path = data_dir()
            .join(round_id)
            .join("analysis")
            .join(format!("ground_truth_seed_index={seed_index}.json"));
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading ground truth {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing ground truth {}", path.display()))
    }
}

impl RoundDetails {
    pub fn load(round_id: &str) -> Result<Self> {
        let path = data_dir().join(round_id).join("details.json");
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading details {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing details {}", path.display()))
    }
}

/// List all round IDs that have analysis data (replays + ground truth).
pub fn list_rounds_with_analysis() -> Result<Vec<String>> {
    let mut rounds = Vec::new();
    for entry in std::fs::read_dir(data_dir())? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let analysis = entry.path().join("analysis");
            if analysis.is_dir() && analysis.join("replay_seed_index=0.json").exists() {
                rounds.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }
    Ok(rounds)
}
```

- [ ] **Step 2: Create world.rs — World state with spatial index**

```rust
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::terrain::TerrainType;
use crate::settlement::Settlement;
use crate::params::Params;
use crate::io::{Grid, ReplayFrame, InitialState};

pub struct World {
    pub width: usize,
    pub height: usize,
    pub grid: Vec<Vec<TerrainType>>,
    pub settlements: Vec<Settlement>,
    pub rng: ChaCha8Rng,
}

impl World {
    /// Create world from a replay frame (for validation — full stats known).
    pub fn from_replay_frame(frame: &ReplayFrame, width: usize, height: usize, seed: u64) -> Self {
        let grid = frame.grid.clone();
        let settlements: Vec<Settlement> = frame.settlements
            .iter()
            .map(Settlement::from_replay)
            .collect();
        Self {
            width,
            height,
            grid,
            settlements,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Create world from initial state (for live inference — stats from params).
    pub fn from_initial_state(initial: &InitialState, params: &Params, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let grid = initial.grid.clone();
        let settlements: Vec<Settlement> = initial.settlements
            .iter()
            .enumerate()
            .map(|(i, is)| {
                let noise = || (rng.random::<f32>() - 0.5) * 0.1;
                Settlement::from_initial(
                    is,
                    (params.init_pop_mean + noise()).max(0.1),
                    (params.init_food_mean + noise()).max(0.0),
                    (params.init_wealth_mean + noise()).max(0.0),
                    (params.init_defense_mean + noise()).max(0.0),
                    i as u16,
                )
            })
            .collect();
        Self { width, height, grid, settlements, rng }
    }

    /// Count adjacent terrain types for a cell (8-connected neighbors).
    pub fn count_adjacent_terrain(&self, x: usize, y: usize) -> TerrainCounts {
        let mut counts = TerrainCounts::default();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }
                match self.grid[ny as usize][nx as usize] {
                    TerrainType::Plains => counts.plains += 1,
                    TerrainType::Forest => counts.forest += 1,
                    TerrainType::Mountain => counts.mountain += 1,
                    TerrainType::Settlement | TerrainType::Port => counts.settlement += 1,
                    TerrainType::Ocean => counts.ocean += 1,
                    _ => {}
                }
            }
        }
        counts
    }

    /// Check if a cell has at least one adjacent ocean tile.
    pub fn has_adjacent_ocean(&self, x: usize, y: usize) -> bool {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < self.width as i32 && ny < self.height as i32 {
                    if self.grid[ny as usize][nx as usize].is_ocean() {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Find alive settlements within Chebyshev distance of a point.
    pub fn settlements_near(&self, x: usize, y: usize, max_dist: f32) -> Vec<usize> {
        let mut result = Vec::new();
        let md = max_dist.ceil() as i32;
        for (i, s) in self.settlements.iter().enumerate() {
            if !s.alive { continue; }
            let dx = (s.x as i32 - x as i32).abs();
            let dy = (s.y as i32 - y as i32).abs();
            if dx <= md && dy <= md {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if dist <= max_dist {
                    result.push(i);
                }
            }
        }
        result
    }

    /// Find settlement at exact position, if any.
    pub fn settlement_at(&self, x: usize, y: usize) -> Option<usize> {
        self.settlements.iter().position(|s| s.alive && s.x == x && s.y == y)
    }

    /// Snapshot the grid as a prediction-class grid (for Monte Carlo aggregation).
    pub fn prediction_snapshot(&self) -> Vec<Vec<usize>> {
        self.grid.iter().map(|row| {
            row.iter().map(|t| t.prediction_class()).collect()
        }).collect()
    }
}

#[derive(Default, Debug, Clone)]
pub struct TerrainCounts {
    pub plains: u8,
    pub forest: u8,
    pub mountain: u8,
    pub settlement: u8,
    pub ocean: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ReplayData;

    #[test]
    fn test_load_replay_and_create_world() {
        // Uses actual data — will skip if data dir doesn't exist
        let rounds = crate::io::list_rounds_with_analysis();
        if rounds.is_err() { return; }
        let rounds = rounds.unwrap();
        if rounds.is_empty() { return; }

        let replay = ReplayData::load(&rounds[0], 0).unwrap();
        let world = World::from_replay_frame(&replay.frames[0], replay.width, replay.height, 42);
        assert_eq!(world.width, 40);
        assert_eq!(world.height, 40);
        assert!(!world.settlements.is_empty());
    }
}
```

- [ ] **Step 3: Update lib.rs**

```rust
pub mod terrain;
pub mod settlement;
pub mod params;
pub mod world;
pub mod io;

pub use terrain::TerrainType;
pub use settlement::Settlement;
pub use params::Params;
pub use world::World;
```

- [ ] **Step 4: Build and run tests**

Run: `cd simulator && cargo test`
Expected: All tests pass. Replay data loads correctly.

- [ ] **Step 5: Commit**

```bash
git add simulator/
git commit -m "feat: add world state, spatial queries, and JSON data loading"
```

---

## Task 3: Simulation Phases (Food, Pop/Defense, Growth)

**Files:**
- Create: `simulator/src/phases/mod.rs`
- Create: `simulator/src/phases/food.rs`
- Create: `simulator/src/phases/pop_defense.rs`
- Create: `simulator/src/phases/growth.rs`
- Modify: `simulator/src/lib.rs`

- [ ] **Step 1: Create phases/mod.rs — Step orchestration**

```rust
pub mod food;
pub mod pop_defense;
pub mod growth;
pub mod conflict;
pub mod trade;
pub mod winter;
pub mod environment;

use crate::world::World;
use crate::params::Params;

/// Run one complete simulation step (one year).
pub fn step(world: &mut World, params: &Params) {
    // Track which cells were ruin BEFORE this step (only these get environment transitions).
    // Ruins created this step (from conflict/winter death) persist until next step.
    let pre_step_ruins: std::collections::HashSet<(usize, usize)> = (0..world.height)
        .flat_map(|y| (0..world.width).filter_map(move |x|
            if world.grid[y][x] == crate::terrain::TerrainType::Ruin { Some((x, y)) } else { None }
        ))
        .collect();

    food::phase_food(world, params);
    pop_defense::phase_pop_defense(world, params);
    growth::phase_growth(world, params);
    conflict::phase_conflict(world, params);
    trade::phase_trade(world, params);
    winter::phase_winter(world, params);
    environment::phase_environment(world, params, &pre_step_ruins);
}

/// Run the full 50-step simulation.
pub fn simulate(world: &mut World, params: &Params, steps: u32) {
    for _ in 0..steps {
        step(world, params);
    }
}
```

- [ ] **Step 2: Create phases/food.rs**

```rust
use crate::world::World;
use crate::params::Params;

/// Phase 1: Food production. Deterministic per settlement based on adjacency.
pub fn phase_food(world: &mut World, params: &Params) {
    for i in 0..world.settlements.len() {
        if !world.settlements[i].alive { continue; }

        let s = &world.settlements[i];
        let counts = world.count_adjacent_terrain(s.x, s.y);

        let food_delta = params.food_base
            + params.food_pop_coeff * s.population
            + params.food_feedback * s.food
            + params.food_plains * counts.plains as f32
            + params.food_forest * counts.forest as f32
            + params.food_mountain * counts.mountain as f32
            + params.food_settlement * counts.settlement as f32;

        world.settlements[i].food = (world.settlements[i].food + food_delta).clamp(0.0, 0.998);
    }
}
```

- [ ] **Step 3: Create phases/pop_defense.rs**

```rust
use crate::world::World;
use crate::params::Params;

/// Phase 2: Population growth (logistic, food-dependent) and defense recovery.
pub fn phase_pop_defense(world: &mut World, params: &Params) {
    for s in world.settlements.iter_mut() {
        if !s.alive { continue; }

        // Logistic population growth
        let pop_delta = params.pop_growth_rate * s.food * (1.0 - s.population / params.pop_max);
        s.population = (s.population + pop_delta).max(0.01);

        // Defense recovery toward baseline (1.0)
        let def_delta = params.defense_recovery_rate * (1.0 - s.defense);
        s.defense = (s.defense + def_delta).clamp(0.0, 1.0);
    }
}
```

- [ ] **Step 4: Create phases/growth.rs**

```rust
use rand::prelude::*;
use crate::terrain::TerrainType;
use crate::settlement::Settlement;
use crate::world::World;
use crate::params::Params;

/// Phase 3: Settlement expansion. Stochastic spawning of new settlements.
pub fn phase_growth(world: &mut World, params: &Params) {
    let n = world.settlements.len();
    let mut new_settlements = Vec::new();

    // Shuffle processing order
    let mut order: Vec<usize> = (0..n).filter(|&i| world.settlements[i].alive).collect();
    order.shuffle(&mut world.rng);

    for &i in &order {
        let s = &world.settlements[i];
        let spawn_prob = params.growth_prob * s.population * s.food;

        if world.rng.random::<f32>() >= spawn_prob { continue; }

        // Find valid target cell (weighted random selection)
        let candidates = find_spawn_candidates(world, s.x, s.y, params);
        if let Some(candidates) = candidates {
            let total_w: f32 = candidates.iter().map(|c| c.3).sum();
            let mut roll = world.rng.random::<f32>() * total_w;
            let mut chosen = &candidates[0];
            for c in &candidates {
                roll -= c.3;
                if roll <= 0.0 { chosen = c; break; }
            }
            let (tx, ty, terrain) = (chosen.0, chosen.1, chosen.2);
        {
            let (child_pop, child_def) = match terrain {
                TerrainType::Ruin => (0.4, 0.15),
                _ => (0.5, 0.2), // Plains or Forest
            };

            // Parent must have enough pop
            if world.settlements[i].population <= child_pop { continue; }
            world.settlements[i].population -= child_pop;

            let is_port = terrain == TerrainType::Ruin && world.has_adjacent_ocean(tx, ty);

            let child = Settlement {
                x: tx,
                y: ty,
                population: child_pop,
                food: world.settlements[i].food * 0.3, // small portion of parent food
                wealth: 0.0,
                defense: child_def,
                has_port: is_port,
                alive: true,
                owner_id: world.settlements[i].owner_id,
            };

            // Update grid
            world.grid[ty][tx] = if is_port { TerrainType::Port } else { TerrainType::Settlement };

            new_settlements.push(child);
        }
    }

    world.settlements.extend(new_settlements);
}

/// Find valid spawn target candidates within distance of parent.
/// Returns list of (x, y, terrain, weight) candidates. Caller uses World.rng to pick.
fn find_spawn_candidates(world: &World, px: usize, py: usize, params: &Params) -> Option<Vec<(usize, usize, TerrainType, f32)>> {
    // Collect buildable cells within spawn distance, weighted by distance
    let max_d = params.spawn_distance_max.ceil() as i32;
    let mut candidates: Vec<(usize, usize, TerrainType, f32)> = Vec::new();

    for dy in -max_d..=max_d {
        for dx in -max_d..=max_d {
            if dx == 0 && dy == 0 { continue; }
            let nx = px as i32 + dx;
            let ny = py as i32 + dy;
            if nx < 0 || ny < 0 || nx >= world.width as i32 || ny >= world.height as i32 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            let terrain = world.grid[ny][nx];
            if !terrain.is_buildable() { continue; }

            // Check no existing alive settlement at this position
            if world.settlement_at(nx, ny).is_some() { continue; }

            let dist = ((dx * dx + dy * dy) as f32).sqrt();
            if dist > params.spawn_distance_max { continue; }

            // Weight: closer cells are much more likely (empirical: 72% dist 1, 21% dist 2)
            let weight = 1.0 / (dist * dist);
            candidates.push((nx, ny, terrain, weight));
        }
    }

    if candidates.is_empty() { return None; }

    Some(candidates)
}
```

- [ ] **Step 5: Create stub files for remaining phases**

Create empty stubs for `conflict.rs`, `trade.rs`, `winter.rs`, `environment.rs` so `mod.rs` compiles:

```rust
// conflict.rs, trade.rs, winter.rs, environment.rs (each file):
use crate::world::World;
use crate::params::Params;

pub fn phase_conflict(world: &mut World, _params: &Params) { /* TODO */ }
// (similar for each)
```

- [ ] **Step 6: Update lib.rs to include phases**

```rust
pub mod terrain;
pub mod settlement;
pub mod params;
pub mod world;
pub mod io;
pub mod phases;

pub use terrain::TerrainType;
pub use settlement::Settlement;
pub use params::Params;
pub use world::World;
```

- [ ] **Step 7: Build and run tests**

Run: `cd simulator && cargo test`
Expected: All tests pass. Phases compile.

- [ ] **Step 8: Commit**

```bash
git add simulator/
git commit -m "feat: add simulation phases — food, pop/defense, growth (+ stubs for remaining)"
```

---

## Task 4: Simulation Phases (Conflict, Trade, Winter, Environment)

**Files:**
- Modify: `simulator/src/phases/conflict.rs`
- Modify: `simulator/src/phases/trade.rs`
- Modify: `simulator/src/phases/winter.rs`
- Modify: `simulator/src/phases/environment.rs`

- [ ] **Step 1: Implement conflict.rs**

```rust
use rand::prelude::*;
use crate::terrain::TerrainType;
use crate::world::World;
use crate::params::Params;

/// Phase 4: Conflict — raiding between different-faction settlements.
pub fn phase_conflict(world: &mut World, params: &Params) {
    let n = world.settlements.len();
    let mut order: Vec<usize> = (0..n).filter(|&i| world.settlements[i].alive).collect();
    order.shuffle(&mut world.rng);

    for &attacker_idx in &order {
        if !world.settlements[attacker_idx].alive { continue; }

        let a = &world.settlements[attacker_idx];
        let ax = a.x;
        let ay = a.y;
        let a_owner = a.owner_id;
        let a_food = a.food;
        let a_port = a.has_port;

        let range = params.raid_range + if a_port { params.longship_range_bonus } else { 0.0 };

        // Find targets within range with different owner
        let nearby = world.settlements_near(ax, ay, range);

        for &defender_idx in &nearby {
            if defender_idx == attacker_idx { continue; }
            if !world.settlements[defender_idx].alive { continue; }
            if world.settlements[defender_idx].owner_id == a_owner { continue; }

            // Raid probability: higher when attacker is food-deprived
            let raid_p = params.raid_prob * (1.0 + params.aggression * (1.0 - a_food).max(0.0));
            if world.rng.random::<f32>() >= raid_p { continue; }

            // Execute raid
            let stolen = world.settlements[defender_idx].wealth * 0.2;
            world.settlements[defender_idx].defense -= params.raid_damage;
            world.settlements[defender_idx].population *= 0.85;
            world.settlements[defender_idx].wealth -= stolen;
            world.settlements[attacker_idx].wealth += stolen;

            // Check for takeover or death
            if world.settlements[defender_idx].defense < params.takeover_threshold {
                if world.rng.random::<f32>() < 0.6 {
                    // Takeover
                    world.settlements[defender_idx].owner_id = a_owner;
                } else {
                    // Death -> ruin
                    let d = &world.settlements[defender_idx];
                    let (dx, dy) = (d.x, d.y);
                    world.settlements[defender_idx].alive = false;
                    world.grid[dy][dx] = TerrainType::Ruin;
                }
            }
        }
    }
}
```

- [ ] **Step 2: Implement trade.rs**

```rust
use crate::world::World;
use crate::params::Params;

/// Phase 5: Trade — allied port pairs within range exchange food and wealth.
pub fn phase_trade(world: &mut World, params: &Params) {
    let n = world.settlements.len();

    // Collect port indices
    let ports: Vec<usize> = (0..n)
        .filter(|&i| world.settlements[i].alive && world.settlements[i].has_port)
        .collect();

    // Check all port pairs
    for i in 0..ports.len() {
        for j in (i + 1)..ports.len() {
            let a = &world.settlements[ports[i]];
            let b = &world.settlements[ports[j]];

            // Must be same faction
            if a.owner_id != b.owner_id { continue; }

            // Must be within trade range
            let dx = (a.x as f32 - b.x as f32);
            let dy = (a.y as f32 - b.y as f32);
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > params.trade_range { continue; }

            // Both gain food and wealth
            world.settlements[ports[i]].food = (world.settlements[ports[i]].food + params.trade_food_bonus).min(0.998);
            world.settlements[ports[j]].food = (world.settlements[ports[j]].food + params.trade_food_bonus).min(0.998);
            world.settlements[ports[i]].wealth += params.trade_wealth_bonus;
            world.settlements[ports[j]].wealth += params.trade_wealth_bonus;
        }
    }
}
```

- [ ] **Step 3: Implement winter.rs**

```rust
use crate::terrain::TerrainType;
use crate::world::World;
use crate::params::Params;

/// Phase 6: Winter — food loss, settlement collapse.
pub fn phase_winter(world: &mut World, params: &Params) {
    for i in 0..world.settlements.len() {
        if !world.settlements[i].alive { continue; }

        world.settlements[i].food -= params.winter_severity;
        world.settlements[i].food = world.settlements[i].food.max(0.0);

        if world.settlements[i].food < params.collapse_threshold {
            // Settlement dies -> ruin
            let (x, y) = (world.settlements[i].x, world.settlements[i].y);
            let owner = world.settlements[i].owner_id;
            let pop = world.settlements[i].population;

            world.settlements[i].alive = false;
            world.grid[y][x] = TerrainType::Ruin;

            // Disperse population to nearby same-faction settlements
            let nearby = world.settlements_near(x, y, 3.0);
            let allies: Vec<usize> = nearby.into_iter()
                .filter(|&j| j != i && world.settlements[j].alive && world.settlements[j].owner_id == owner)
                .collect();
            if !allies.is_empty() {
                let share = pop / allies.len() as f32;
                for &j in &allies {
                    world.settlements[j].population += share;
                }
            }
        }
    }
}
```

- [ ] **Step 4: Implement environment.rs**

```rust
use rand::prelude::*;
use crate::terrain::TerrainType;
use crate::settlement::Settlement;
use crate::world::World;
use crate::params::Params;

/// Phase 7: Environment — ruin transitions (settlement/forest/plains/port).
/// Only processes ruins that existed BEFORE this step (ruins last exactly 1 step).
pub fn phase_environment(world: &mut World, params: &Params, pre_step_ruins: &std::collections::HashSet<(usize, usize)>) {
    let mut new_settlements = Vec::new();

    for y in 0..world.height {
        for x in 0..world.width {
            if world.grid[y][x] != TerrainType::Ruin { continue; }
            // Only transition ruins that existed before this step
            if !pre_step_ruins.contains(&(x, y)) { continue; }

            let nearby = world.settlements_near(x, y, params.reclaim_range);
            let has_nearby_settlement = !nearby.is_empty();
            let has_ocean = world.has_adjacent_ocean(x, y);

            let p_settlement = if has_nearby_settlement { params.ruin_to_settlement } else { 0.0 };
            let p_port = if has_nearby_settlement && has_ocean { params.ruin_to_settlement * 0.1 } else { 0.0 };
            let p_forest = params.ruin_to_forest;
            let p_plains = (1.0 - p_settlement - p_port - p_forest).max(0.0);

            let roll = world.rng.random::<f32>();

            if roll < p_settlement {
                // Reclaim as settlement
                let nearest = nearby[0]; // nearest alive settlement
                let parent = &world.settlements[nearest];
                let new_s = Settlement {
                    x, y,
                    population: 0.4,
                    food: parent.food * 0.2,
                    wealth: 0.0,
                    defense: 0.15,
                    has_port: false,
                    alive: true,
                    owner_id: parent.owner_id,
                };
                world.grid[y][x] = TerrainType::Settlement;
                new_settlements.push(new_s);
            } else if roll < p_settlement + p_port {
                // Reclaim as port
                let nearest = nearby[0];
                let parent = &world.settlements[nearest];
                let new_s = Settlement {
                    x, y,
                    population: 0.4,
                    food: parent.food * 0.2,
                    wealth: 0.0,
                    defense: 0.15,
                    has_port: true,
                    alive: true,
                    owner_id: parent.owner_id,
                };
                world.grid[y][x] = TerrainType::Port;
                new_settlements.push(new_s);
            } else if roll < p_settlement + p_port + p_forest {
                world.grid[y][x] = TerrainType::Forest;
            } else {
                world.grid[y][x] = TerrainType::Plains;
            }
        }
    }

    world.settlements.extend(new_settlements);
}
```

- [ ] **Step 5: Build and test**

Run: `cd simulator && cargo test`
Expected: All tests pass, all phases compile.

- [ ] **Step 6: Commit**

```bash
git add simulator/
git commit -m "feat: implement conflict, trade, winter, and environment phases"
```

---

## Task 5: Replay Validation Test

**Files:**
- Create: `simulator/tests/replay_validation.rs`

This is the critical validation: run our simulator on replay step 0 and compare frame-by-frame against the actual replay data.

- [ ] **Step 1: Write replay validation test**

```rust
//! Integration test: compare simulator output against actual replay data.

use astar_simulator::io::{ReplayData, list_rounds_with_analysis};
use astar_simulator::world::World;
use astar_simulator::params::Params;
use astar_simulator::phases;

/// Compare terrain grids, return fraction of matching cells.
fn grid_match_fraction(a: &[Vec<astar_simulator::TerrainType>], b: &[Vec<astar_simulator::TerrainType>]) -> f64 {
    let mut match_count = 0usize;
    let mut total = 0usize;
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (ca, cb) in row_a.iter().zip(row_b.iter()) {
            // Compare prediction classes (not internal codes)
            if ca.prediction_class() == cb.prediction_class() {
                match_count += 1;
            }
            total += 1;
        }
    }
    match_count as f64 / total as f64
}

/// Count settlements by alive status.
fn count_alive(world: &World) -> usize {
    world.settlements.iter().filter(|s| s.alive).count()
}

#[test]
fn test_single_step_replay_fidelity() {
    let rounds = list_rounds_with_analysis().expect("need data dir");
    if rounds.is_empty() { return; }

    let replay = ReplayData::load(&rounds[0], 0).expect("load replay");
    let params = Params::default_prior();

    // Start from replay frame 0 (exact state)
    let mut world = World::from_replay_frame(&replay.frames[0], replay.width, replay.height, 42);

    let initial_alive = count_alive(&world);

    // Run one step
    phases::step(&mut world, &params);

    // With default params, we don't expect perfect match,
    // but grid should be mostly similar (>70% same terrain classes)
    let match_frac = grid_match_fraction(&world.grid, &replay.frames[1].grid);
    println!("Step 0->1 grid match: {:.1}%", match_frac * 100.0);

    // Sanity: settlements should still exist
    let alive = count_alive(&world);
    println!("Alive settlements: {} -> {}", initial_alive, alive);
    assert!(alive > 0, "all settlements died in one step");
}

#[test]
fn test_full_simulation_runs_without_panic() {
    let rounds = list_rounds_with_analysis().expect("need data dir");
    if rounds.is_empty() { return; }

    let replay = ReplayData::load(&rounds[0], 0).expect("load replay");
    let params = Params::default_prior();

    let mut world = World::from_replay_frame(&replay.frames[0], replay.width, replay.height, 42);
    phases::simulate(&mut world, &params, 50);

    // Should not panic; settlements should exist
    let alive = count_alive(&world);
    println!("After 50 steps: {} alive settlements", alive);
    // With default params this is very rough; the point is no panics
}
```

- [ ] **Step 2: Run the test**

Run: `cd simulator && cargo test --test replay_validation -- --nocapture`
Expected: Both tests pass. Grid match will be rough with default params (that's expected — we'll tune later).

- [ ] **Step 3: Commit**

```bash
git add simulator/
git commit -m "test: add replay validation integration tests"
```

---

## Task 6: Monte Carlo Runner + Scoring

**Files:**
- Create: `simulator/src/montecarlo.rs`
- Create: `simulator/src/scoring.rs`
- Modify: `simulator/src/lib.rs`

- [ ] **Step 1: Create montecarlo.rs**

```rust
use rayon::prelude::*;
use crate::io::{InitialState, Grid};
use crate::params::Params;
use crate::world::World;
use crate::phases;

/// Run N simulations and aggregate terrain class frequencies into a probability distribution.
/// Returns: height x width x 6 probability tensor.
pub fn run_montecarlo(
    initial_state: &InitialState,
    params: &Params,
    n_sims: usize,
    steps: u32,
    base_seed: u64,
) -> Vec<Vec<[f64; 6]>> {
    let height = initial_state.grid.len();
    let width = initial_state.grid[0].len();

    // Run simulations in parallel
    let snapshots: Vec<Vec<Vec<usize>>> = (0..n_sims)
        .into_par_iter()
        .map(|i| {
            let seed = base_seed.wrapping_add(i as u64);
            let mut world = World::from_initial_state(initial_state, params, seed);
            phases::simulate(&mut world, params, steps);
            world.prediction_snapshot()
        })
        .collect();

    // Aggregate counts
    let mut counts = vec![vec![[0u32; 6]; width]; height];
    for snap in &snapshots {
        for y in 0..height {
            for x in 0..width {
                counts[y][x][snap[y][x]] += 1;
            }
        }
    }

    // Normalize to probabilities with floor
    let floor = 0.01_f64;
    let n = n_sims as f64;

    counts.iter().map(|row| {
        row.iter().map(|cell| {
            let mut probs: [f64; 6] = std::array::from_fn(|c| (cell[c] as f64 / n).max(floor));
            let sum: f64 = probs.iter().sum();
            for p in &mut probs { *p /= sum; }
            probs
        }).collect()
    }).collect()
}
```

- [ ] **Step 2: Create scoring.rs**

```rust
/// Compute entropy-weighted KL divergence score between prediction and ground truth.
/// Higher score = better prediction (score is inverted from raw KL).
pub fn score_prediction(
    prediction: &[Vec<[f64; 6]>],
    ground_truth: &[Vec<[f64; 6]>],
) -> f64 {
    let height = ground_truth.len();
    let width = ground_truth[0].len();

    let mut total_weighted_kl = 0.0_f64;
    let mut total_weight = 0.0_f64;

    for y in 0..height {
        for x in 0..width {
            let gt = &ground_truth[y][x];
            let pred = &prediction[y][x];

            // Compute entropy of ground truth
            let entropy: f64 = gt.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();

            // Skip near-zero entropy cells (static terrain)
            if entropy < 1e-6 { continue; }

            // KL divergence: sum p * ln(p/q)
            let kl: f64 = gt.iter().zip(pred.iter())
                .filter(|(&p, _)| p > 0.0)
                .map(|(&p, &q)| {
                    let q_safe = q.max(1e-10);
                    p * (p / q_safe).ln()
                })
                .sum();

            total_weighted_kl += entropy * kl;
            total_weight += entropy;
        }
    }

    if total_weight < 1e-10 { return 0.0; }

    // Score: lower KL = better. Return negative weighted mean KL so higher = better.
    // The competition likely uses a transformation; for now return raw weighted mean KL.
    total_weighted_kl / total_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_prediction_has_zero_kl() {
        let gt = vec![vec![[0.5, 0.3, 0.0, 0.1, 0.1, 0.0]]];
        let pred = gt.clone();
        let score = score_prediction(&pred, &gt);
        assert!(score.abs() < 1e-10, "perfect prediction should have 0 KL, got {score}");
    }

    #[test]
    fn test_uniform_prediction_worse_than_informed() {
        let gt = vec![vec![[0.0, 0.8, 0.0, 0.1, 0.1, 0.0]]];
        let good = vec![vec![[0.01, 0.78, 0.01, 0.1, 0.09, 0.01]]];
        let uniform = vec![vec![[1.0/6.0; 6]]];

        let score_good = score_prediction(&good, &gt);
        let score_unif = score_prediction(&uniform, &gt);
        assert!(score_good < score_unif, "informed prediction should have lower KL");
    }
}
```

- [ ] **Step 3: Update lib.rs**

Add `pub mod montecarlo;` and `pub mod scoring;`.

- [ ] **Step 4: Build and test**

Run: `cd simulator && cargo test`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add simulator/
git commit -m "feat: add Monte Carlo runner with Rayon parallelism and KL divergence scoring"
```

---

## Task 7: Oracle Validation Against Ground Truth

**Files:**
- Create: `simulator/tests/oracle_validation.rs`

Run Monte Carlo with default params from replay initial state, compare to ground truth. This tells us how close our simulator is.

- [ ] **Step 1: Write oracle validation test**

```rust
use astar_simulator::io::{ReplayData, GroundTruthData, InitialState, InitialSettlement, list_rounds_with_analysis};
use astar_simulator::montecarlo::run_montecarlo;
use astar_simulator::scoring::score_prediction;
use astar_simulator::params::Params;

#[test]
fn test_oracle_montecarlo_vs_ground_truth() {
    let rounds = list_rounds_with_analysis().expect("need data dir");
    if rounds.is_empty() { return; }

    let round_id = &rounds[0];
    let replay = ReplayData::load(round_id, 0).expect("load replay");
    let gt = GroundTruthData::load(round_id, 0).expect("load ground truth");
    let params = Params::default_prior();

    // Build InitialState from replay frame 0 (oracle: we know exact initial grid)
    let initial = InitialState {
        grid: replay.frames[0].grid.clone(),
        settlements: replay.frames[0].settlements.iter()
            .map(|s| InitialSettlement {
                x: s.x, y: s.y, has_port: s.has_port, alive: s.alive,
            })
            .collect(),
    };

    // Run Monte Carlo: 100 sims (quick for testing)
    let prediction = run_montecarlo(&initial, &params, 100, 50, 42);
    let kl = score_prediction(&prediction, &gt.ground_truth);
    println!("Oracle KL divergence (100 sims, default params): {kl:.4}");

    assert!(kl.is_finite(), "KL should be finite");
    assert!(kl >= 0.0, "KL should be non-negative");
}
```

- [ ] **Step 2: Run oracle test**

Run: `cd simulator && cargo test --test oracle_validation -- --nocapture`
Expected: KL score is printed. It will be imperfect with default params — that's fine. The goal is to validate the pipeline works end-to-end.

- [ ] **Step 3: Commit**

```bash
git add simulator/
git commit -m "test: add oracle validation — Monte Carlo vs ground truth"
```

---

## Task 8: Evolution Strategy Parameter Inference

**Files:**
- Create: `simulator/src/inference.rs`
- Modify: `simulator/src/lib.rs`
- Modify: `simulator/Cargo.toml`

- [ ] **Step 1: Add CMA-ES dependency**

We implement a truncation-selection evolution strategy (ES) with per-dimension adaptive sigma. This is simpler than full CMA-ES but converges well for ~30 parameters. Can be upgraded to full CMA-ES later if needed.

- [ ] **Step 2: Implement inference.rs with evolution strategy**

```rust
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::io::{InitialState, QueryResult};
use crate::params::Params;
use crate::world::World;
use crate::phases;
use crate::terrain::TerrainType;

/// Viewport observation: seed index + viewport bounds + observed terrain grid.
pub struct Observation {
    pub seed_index: usize,
    pub vx: usize,
    pub vy: usize,
    pub vw: usize,
    pub vh: usize,
    pub grid: Vec<Vec<TerrainType>>,
}

/// Evolution strategy configuration.
pub struct InferenceConfig {
    pub n_generations: usize,
    pub population_size: usize,
    pub sims_per_candidate: usize,
    pub steps: u32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_generations: 200,
            population_size: 50,
            sims_per_candidate: 20,
            steps: 50,
        }
    }
}

/// Run evolution strategy to infer hidden parameters from observations.
/// Returns best params found.
pub fn infer_params(
    initial_states: &[InitialState],   // one per seed
    observations: &[Observation],       // viewport observations
    config: &InferenceConfig,
) -> Params {
    let dim = 30;
    let prior = Params::default_prior();
    let mut mean = prior.to_vec();
    let lower = Params::lower_bounds();
    let upper = Params::upper_bounds();

    // Initialize sigma (step size) per dimension
    let mut sigma: Vec<f32> = mean.iter().enumerate().map(|(i, &m)| {
        (upper[i] - lower[i]) * 0.2 // 20% of range
    }).collect();

    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = mean.clone();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for gen in 0..config.n_generations {
        // Generate population: mean + gaussian noise * sigma
        let candidates: Vec<Vec<f32>> = (0..config.population_size)
            .map(|_| {
                mean.iter().enumerate().map(|(i, &m)| {
                    // Box-Muller for approximate Gaussian noise
                let u1: f32 = rng.random::<f32>().max(1e-10);
                let u2: f32 = rng.random::<f32>();
                let noise: f32 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    let v = m + noise * sigma[i];
                    v.clamp(lower[i], upper[i])
                }).collect()
            })
            .collect();

        // Evaluate all candidates in parallel
        let scores: Vec<f64> = candidates.par_iter()
            .map(|candidate| {
                let params = Params::from_vec(candidate);
                evaluate_params(&params, initial_states, observations, config)
            })
            .collect();

        // Sort by score (higher is better = less KL)
        let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Update best
        if indexed[0].1 > best_score {
            best_score = indexed[0].1;
            best_params = candidates[indexed[0].0].clone();
        }

        // Update mean: weighted average of top 25%
        let elite_count = (config.population_size / 4).max(1);
        let mut new_mean = vec![0.0f32; dim];
        for &(idx, _) in &indexed[..elite_count] {
            for (j, &v) in candidates[idx].iter().enumerate() {
                new_mean[j] += v / elite_count as f32;
            }
        }

        // Update sigma: standard deviation of elites
        for j in 0..dim {
            let variance: f32 = indexed[..elite_count].iter()
                .map(|&(idx, _)| {
                    let diff = candidates[idx][j] - new_mean[j];
                    diff * diff
                })
                .sum::<f32>() / elite_count as f32;
            sigma[j] = variance.sqrt().max(0.001); // minimum sigma
        }

        mean = new_mean;

        if gen % 20 == 0 {
            eprintln!("Gen {gen}: best_score={best_score:.4}, mean_top={:.4}",
                indexed[..elite_count].iter().map(|x| x.1).sum::<f64>() / elite_count as f64);
        }
    }

    Params::from_vec(&best_params)
}

/// Evaluate a parameter set: run simulations, compare to observations.
/// Returns negative mean log-likelihood (higher = better match).
fn evaluate_params(
    params: &Params,
    initial_states: &[InitialState],
    observations: &[Observation],
    config: &InferenceConfig,
) -> f64 {
    let mut total_ll = 0.0_f64;
    let mut total_cells = 0usize;

    // Group observations by seed
    let max_seed = observations.iter().map(|o| o.seed_index).max().unwrap_or(0);

    for seed_idx in 0..=max_seed {
        let seed_obs: Vec<&Observation> = observations.iter()
            .filter(|o| o.seed_index == seed_idx)
            .collect();
        if seed_obs.is_empty() { continue; }
        if seed_idx >= initial_states.len() { continue; }

        // Run K simulations for this seed
        let snapshots: Vec<Vec<Vec<usize>>> = (0..config.sims_per_candidate)
            .map(|i| {
                let seed = (seed_idx as u64 * 10000) + i as u64;
                let mut world = World::from_initial_state(&initial_states[seed_idx], params, seed);
                phases::simulate(&mut world, params, config.steps);
                world.prediction_snapshot()
            })
            .collect();

        // For each observation viewport, compute log-likelihood
        for obs in &seed_obs {
            for vy in 0..obs.vh {
                for vx in 0..obs.vw {
                    let map_y = obs.vy + vy;
                    let map_x = obs.vx + vx;
                    let observed_class = obs.grid[vy][vx].prediction_class();

                    // Count how many simulations produced this terrain class at this cell
                    let count: usize = snapshots.iter()
                        .filter(|snap| snap[map_y][map_x] == observed_class)
                        .count();

                    // Log-likelihood with floor
                    let prob = ((count as f64) / (config.sims_per_candidate as f64)).max(0.01);
                    total_ll += prob.ln();
                    total_cells += 1;
                }
            }
        }
    }

    if total_cells == 0 { return f64::NEG_INFINITY; }
    total_ll / total_cells as f64
}
```

- [ ] **Step 3: Update lib.rs**

Add `pub mod inference;`.

- [ ] **Step 4: Build and test**

Run: `cd simulator && cargo test`
Expected: Compiles. No runtime test for inference yet (it's expensive).

- [ ] **Step 5: Commit**

```bash
git add simulator/
git commit -m "feat: add CMA-ES parameter inference from viewport observations"
```

---

## Task 9: CLI Entry Point

**Files:**
- Create: `simulator/src/main.rs`

- [ ] **Step 1: Create main.rs with subcommands**

```rust
use std::process;
use anyhow::Result;

use astar_simulator::io::{self, ReplayData, GroundTruthData, RoundDetails};
use astar_simulator::params::Params;
use astar_simulator::montecarlo::run_montecarlo;
use astar_simulator::scoring::score_prediction;
use astar_simulator::inference::{self, Observation, InferenceConfig};
use astar_simulator::world::World;
use astar_simulator::phases;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: astar-simulator <command> [args...]");
        eprintln!("Commands:");
        eprintln!("  oracle <round_id> <seed> <n_sims>   — Run Monte Carlo from replay, score vs GT");
        eprintln!("  montecarlo <round_id> <n_sims>      — Run Monte Carlo for all seeds, output JSON");
        eprintln!("  infer <round_id> <queries_json>      — Infer params from query observations");
        eprintln!("  validate-all <n_sims>                — Oracle test on all available rounds");
        process::exit(1);
    }

    match args[1].as_str() {
        "oracle" => cmd_oracle(&args[2..])?,
        "montecarlo" => cmd_montecarlo(&args[2..])?,
        "infer" => cmd_infer(&args[2..])?,
        "validate-all" => cmd_validate_all(&args[2..])?,
        other => {
            eprintln!("Unknown command: {other}");
            process::exit(1);
        }
    }

    Ok(())
}

fn cmd_oracle(args: &[String]) -> Result<()> {
    let round_id = &args[0];
    let seed: usize = args[1].parse()?;
    let n_sims: usize = args[2].parse()?;

    let replay = ReplayData::load(round_id, seed)?;
    let gt = GroundTruthData::load(round_id, seed)?;
    let params = Params::default_prior();

    // Build initial state from replay frame 0
    let initial = io::InitialState {
        grid: replay.frames[0].grid.clone(),
        settlements: replay.frames[0].settlements.iter()
            .map(|s| io::InitialSettlement {
                x: s.x, y: s.y, has_port: s.has_port, alive: s.alive,
            })
            .collect(),
    };

    eprintln!("Running {n_sims} simulations for round {round_id} seed {seed}...");
    let prediction = run_montecarlo(&initial, &params, n_sims, 50, 42);
    let kl = score_prediction(&prediction, &gt.ground_truth);
    println!("KL divergence: {kl:.6}");

    Ok(())
}

fn cmd_montecarlo(args: &[String]) -> Result<()> {
    let round_id = &args[0];
    let n_sims: usize = args[1].parse()?;

    let details = RoundDetails::load(round_id)?;
    let params = Params::default_prior();

    for seed_idx in 0..details.seeds_count {
        let initial = &details.initial_states[seed_idx];
        eprintln!("Seed {seed_idx}: running {n_sims} simulations...");
        let prediction = run_montecarlo(initial, &params, n_sims, 50, seed_idx as u64 * 10000);

        // Output as JSON
        let json = serde_json::to_string(&prediction)?;
        println!("SEED_{seed_idx}:{json}");
    }

    Ok(())
}

fn cmd_validate_all(args: &[String]) -> Result<()> {
    let n_sims: usize = args.get(0).map(|s| s.parse().unwrap()).unwrap_or(100);

    let rounds = io::list_rounds_with_analysis()?;
    eprintln!("Found {} rounds with analysis data", rounds.len());

    for round_id in &rounds {
        let details = RoundDetails::load(round_id)?;
        let params = Params::default_prior();

        let mut round_kl = 0.0;
        let mut count = 0;

        for seed_idx in 0..details.seeds_count.min(5) {
            let gt = match GroundTruthData::load(round_id, seed_idx) {
                Ok(gt) => gt,
                Err(_) => continue,
            };

            let initial = &details.initial_states[seed_idx];
            let prediction = run_montecarlo(initial, &params, n_sims, 50, seed_idx as u64 * 10000);
            let kl = score_prediction(&prediction, &gt.ground_truth);
            eprintln!("Round {} seed {}: KL={:.4}", details.round_number, seed_idx, kl);
            round_kl += kl;
            count += 1;
        }

        if count > 0 {
            eprintln!("Round {} average KL: {:.4}", details.round_number, round_kl / count as f64);
        }
    }

    Ok(())
}

fn cmd_infer(args: &[String]) -> Result<()> {
    let round_id = &args[0];
    let query_dir = io::data_dir().join(round_id).join("query");

    let details = RoundDetails::load(round_id)?;

    // Load query results as observations
    let mut observations = Vec::new();
    for entry in std::fs::read_dir(&query_dir)? {
        let entry = entry?;
        let raw = std::fs::read_to_string(entry.path())?;
        let qr: io::QueryResult = serde_json::from_str(&raw)?;

        // Parse seed_index from filename (map_idx=N_...)
        let fname = entry.file_name().to_string_lossy().to_string();
        let seed_idx: usize = fname.split("map_idx=").nth(1)
            .and_then(|s| s.split('_').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        observations.push(Observation {
            seed_index: seed_idx,
            vx: qr.viewport.x,
            vy: qr.viewport.y,
            vw: qr.viewport.w,
            vh: qr.viewport.h,
            grid: qr.grid,
        });
    }

    eprintln!("Loaded {} observations for round {round_id}", observations.len());

    let config = InferenceConfig::default();
    let best_params = inference::infer_params(&details.initial_states, &observations, &config);

    // Output inferred params as JSON
    let params_vec = best_params.to_vec();
    println!("{}", serde_json::to_string(&params_vec)?);

    Ok(())
}
```

- [ ] **Step 2: Build release binary**

Run: `cd simulator && cargo build --release`
Expected: Binary at `simulator/target/release/astar-simulator`.

- [ ] **Step 3: Run validate-all with 100 sims**

Run: `cd simulator && cargo run --release -- validate-all 100`
Expected: Prints KL scores for all rounds. Scores will be rough but finite.

- [ ] **Step 4: Commit**

```bash
git add simulator/
git commit -m "feat: add CLI entry point — oracle, montecarlo, validate-all commands"
```

---

## Task 10: Python Orchestrator

**Files:**
- Create: `src/astar_island/orchestrator.py`
- Create: `src/astar_island/query_planner.py`

- [ ] **Step 1: Create query_planner.py**

```python
"""Strategic viewport placement for 50 queries across 5 seeds."""

def plan_queries(initial_states: list[dict], budget: int = 50) -> list[dict]:
    """
    Plan viewport queries to maximize information for parameter inference.

    Returns list of {seed_index, viewport_x, viewport_y, viewport_w, viewport_h}.
    """
    queries = []

    # Strategy: 2 primary seeds get 18 queries each, 3 secondary get ~5 each
    # Primary seeds: full coverage (9 viewports tiling 40x40) + 9 repeats
    # Secondary seeds: 5 viewports covering settlement-rich areas

    seed_budgets = [18, 18, 5, 5, 4]

    # Tiling positions for full coverage of 40x40 with 15x15 viewports
    # API constraint: viewport_x and viewport_y must be < 25
    tile_positions = [
        (0, 0), (0, 13), (0, 24),
        (13, 0), (13, 13), (13, 24),
        (24, 0), (24, 13), (24, 24),
    ]

    for seed_idx, budget_for_seed in enumerate(seed_budgets):
        if budget_for_seed >= 9:
            # Full coverage + repeats
            for vy, vx in tile_positions:
                queries.append({
                    "seed_index": seed_idx,
                    "viewport_x": vx,
                    "viewport_y": vy,
                    "viewport_w": min(15, 40 - vx),
                    "viewport_h": min(15, 40 - vy),
                })
            # Repeat center viewports for stochastic variance
            repeats_left = budget_for_seed - 9
            for r in range(repeats_left):
                pos = tile_positions[r % len(tile_positions)]
                queries.append({
                    "seed_index": seed_idx,
                    "viewport_x": pos[1],
                    "viewport_y": pos[0],
                    "viewport_w": min(15, 40 - pos[1]),
                    "viewport_h": min(15, 40 - pos[0]),
                })
        else:
            # Partial coverage — center + settlement-rich areas
            # Use initial_states to find settlement positions
            settlements = initial_states[seed_idx].get("settlements", [])
            if settlements:
                # Cluster settlements and pick viewports to cover clusters
                xs = [s["x"] for s in settlements]
                ys = [s["y"] for s in settlements]
                cx, cy = int(sum(xs)/len(xs)), int(sum(ys)/len(ys))
                # Center viewport on settlement centroid
                vx = max(0, min(cx - 7, 25))
                vy = max(0, min(cy - 7, 25))
                queries.append({
                    "seed_index": seed_idx,
                    "viewport_x": vx,
                    "viewport_y": vy,
                    "viewport_w": 15,
                    "viewport_h": 15,
                })
                # Add more viewports for remaining budget
                for r in range(budget_for_seed - 1):
                    queries.append({
                        "seed_index": seed_idx,
                        "viewport_x": vx,
                        "viewport_y": vy,
                        "viewport_w": 15,
                        "viewport_h": 15,
                    })

    return queries[:budget]
```

- [ ] **Step 2: Create orchestrator.py**

```python
"""End-to-end pipeline: fetch round, run queries, infer params, predict, submit."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

from astar_island.data.client import (
    BASE_API_URL,
    auth_cookies,
    get_active_round_id,
    get_round_details,
    get_simulation_result,
    round_data_path,
)
from astar_island.query_planner import plan_queries

SIMULATOR_BIN = Path(__file__).parent.parent.parent / "simulator" / "target" / "release" / "astar-simulator"


def submit_prediction(round_id: str, seed_index: int, prediction: list):
    """Submit a prediction tensor for one seed."""
    resp = requests.post(
        f"{BASE_API_URL}/astar-island/submit",
        json={
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        },
        cookies=auth_cookies,
    )
    resp.raise_for_status()
    return resp.json()


def run_pipeline(round_id: str | None = None, n_sims: int = 2000):
    """Full pipeline: fetch, query, infer, predict, submit."""
    if round_id is None:
        round_id = get_active_round_id()

    details = get_round_details(round_id)
    print(f"Round {details['round_number']} ({round_id})")
    print(f"Map: {details['map_width']}x{details['map_height']}, {details['seeds_count']} seeds")

    # Step 1: Plan and execute queries
    queries = plan_queries(details["initial_states"])
    print(f"Planned {len(queries)} queries")

    query_results = []
    for q in queries:
        result = get_simulation_result(
            round_id,
            map_idx=q["seed_index"],
            r=q["viewport_y"],
            c=q["viewport_x"],
            run_seed_idx=len(query_results),
        )
        query_results.append(result)

    # Step 2: Run Monte Carlo via Rust binary
    print(f"Running Monte Carlo ({n_sims} sims per seed)...")
    result = subprocess.run(
        [str(SIMULATOR_BIN), "montecarlo", round_id, str(n_sims)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )

    if result.returncode != 0:
        print(f"Simulator error: {result.stderr}")
        return

    # Parse output: SEED_0:{json}, SEED_1:{json}, ...
    predictions = {}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("SEED_"):
            parts = line.split(":", 1)
            seed_idx = int(parts[0].split("_")[1])
            predictions[seed_idx] = json.loads(parts[1])

    # Step 3: Submit predictions
    for seed_idx in range(details["seeds_count"]):
        if seed_idx in predictions:
            print(f"Submitting seed {seed_idx}...")
            resp = submit_prediction(round_id, seed_idx, predictions[seed_idx])
            print(f"  Response: {resp}")
        else:
            print(f"  WARNING: No prediction for seed {seed_idx}")


if __name__ == "__main__":
    import sys
    round_id = sys.argv[1] if len(sys.argv) > 1 else None
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    run_pipeline(round_id, n_sims)
```

- [ ] **Step 3: Commit**

```bash
git add src/astar_island/orchestrator.py src/astar_island/query_planner.py
git commit -m "feat: add Python orchestrator and query planner"
```

---

## Task 11: Iterative Calibration Against Replays

This is not a code task — it's the critical tuning phase where we run the simulator against all 9 rounds of replay data and iteratively adjust the phase logic until frame-by-frame fidelity is high.

**Files:**
- Modify: Various phase files based on calibration results

- [ ] **Step 1: Run validate-all with 500 sims**

Run: `cd simulator && cargo run --release -- validate-all 500`
Record KL scores per round.

- [ ] **Step 2: Analyze which rounds/seeds have worst KL**

Look at which terrain classes are most wrong. Common issues:
- Too many/few settlements → adjust `growth_prob`, `collapse_threshold`
- Too many/few ruins → adjust `winter_severity`, `raid_damage`
- Wrong port count → adjust port formation logic

- [ ] **Step 3: Tune parameters and re-validate**

Iterate: adjust default_prior values → run validate-all → compare scores. Focus on the food model coefficients first (most constrained by data), then growth/conflict balance.

- [ ] **Step 4: Test phase ordering alternatives**

Try food production inside the growth phase instead of as a separate Phase 1. Compare frame-by-frame match rates.

- [ ] **Step 5: Commit tuned parameters**

```bash
git add simulator/
git commit -m "tune: calibrate default params against replay data"
```

---

## Task 12: End-to-End Integration Test

**Files:**
- Create: `simulator/tests/e2e.rs`

- [ ] **Step 1: Write E2E test that uses infer + montecarlo + score**

Use a historical round: take the initial states + query observations, run inference, run Monte Carlo, score against ground truth.

- [ ] **Step 2: Run and verify score is reasonable**

Run: `cd simulator && cargo test --test e2e --release -- --nocapture`
Expected: Score should be meaningfully better than uniform baseline.

- [ ] **Step 3: Commit**

```bash
git add simulator/
git commit -m "test: end-to-end integration test — infer + montecarlo + score"
```

---

## Task 13: Performance Optimization

**Files:**
- Modify: `simulator/src/world.rs` (spatial index)
- Modify: `simulator/src/phases/conflict.rs` (use spatial index)

- [ ] **Step 1: Add grid-based spatial index to World**

Instead of scanning all settlements for `settlements_near`, maintain a `HashMap<(usize, usize), Vec<usize>>` mapping grid cells to settlement indices. Update on settlement creation/death.

- [ ] **Step 2: Benchmark: time 10K simulations**

Run: `cd simulator && cargo run --release -- oracle <round_id> 0 10000`
Record wall time. Target: <15 seconds for 10K sims on a single machine.

- [ ] **Step 3: Profile and optimize hotspots**

Use `cargo flamegraph` or manual timing to find bottlenecks. Common targets:
- Reduce allocations in per-step loops (reuse buffers)
- Avoid Vec cloning in grid operations
- Use `SmallVec` for settlement lists

- [ ] **Step 4: Commit optimizations**

```bash
git add simulator/
git commit -m "perf: add spatial index and optimize simulation throughput"
```
