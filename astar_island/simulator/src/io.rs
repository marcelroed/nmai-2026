use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

use crate::settlement::{InitialSettlement, ReplaySettlement};
use crate::terrain::TerrainType;

// ── Type aliases ─────────────────────────────────────────────────────────────

/// A 2-D terrain grid: `grid[row][col]`.
pub type Grid = Vec<Vec<TerrainType>>;

/// A 2-D distribution grid: `dist[row][col]` is a 6-element probability
/// vector over the prediction classes.
pub type DistributionGrid = Vec<Vec<[f64; 6]>>;

// ── Data directory resolution ─────────────────────────────────────────────────

/// Resolve the data directory.
///
/// Priority:
/// 1. `ASTAR_DATA_DIR` environment variable.
/// 2. `"../data"` relative to the current working directory.
/// 3. `"data"` relative to the current working directory.
/// 4. `"astar_island/data"` relative to the current working directory.
///
/// Panics with a helpful message if none of the above paths exists.
pub fn data_dir() -> PathBuf {
    // 1. Environment variable override.
    if let Ok(val) = std::env::var("ASTAR_DATA_DIR") {
        let p = PathBuf::from(val);
        if p.is_dir() {
            return p;
        }
        panic!(
            "ASTAR_DATA_DIR is set to {:?} but that directory does not exist",
            p
        );
    }

    // 2-4. Fallback candidates.
    let candidates = ["../data", "data", "astar_island/data"];
    for candidate in &candidates {
        let p = PathBuf::from(candidate);
        if p.is_dir() {
            return p;
        }
    }

    panic!(
        "Could not find the data directory. \
         Set the ASTAR_DATA_DIR environment variable to its absolute path, \
         or place the data at one of: {:?}",
        candidates
    );
}

// ── Replay data ───────────────────────────────────────────────────────────────

/// A single timestep captured in a replay file.
#[derive(Debug, Deserialize)]
pub struct ReplayFrame {
    pub step: u32,
    pub grid: Grid,
    pub settlements: Vec<ReplaySettlement>,
}

/// Full replay for one (round, seed) pair.
#[derive(Debug, Deserialize)]
pub struct ReplayData {
    pub round_id: String,
    pub seed_index: usize,
    pub sim_seed: u64,
    pub width: usize,
    pub height: usize,
    pub frames: Vec<ReplayFrame>,
}

impl ReplayData {
    /// Load replay from
    /// `<data_dir>/<round_id>/analysis/replay_seed_index=<seed_index>.json`.
    pub fn load(round_id: &str, seed_index: usize) -> Result<Self> {
        let path = data_dir()
            .join(round_id)
            .join("analysis")
            .join(format!("replay_seed_index={}.json", seed_index));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading replay from {:?}", path))?;
        let data: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing replay JSON from {:?}", path))?;
        Ok(data)
    }
}

// ── Ground truth data ─────────────────────────────────────────────────────────

/// Ground-truth (and optional prediction/score) for one (round, seed) pair.
#[derive(Debug, Deserialize)]
pub struct GroundTruthData {
    pub width: usize,
    pub height: usize,
    pub initial_grid: Grid,
    pub ground_truth: DistributionGrid,
    /// Submitted prediction, if any (may be absent or JSON null).
    pub prediction: Option<DistributionGrid>,
    /// Score awarded, if any (may be absent or JSON null).
    pub score: Option<f64>,
}

impl GroundTruthData {
    /// Load ground truth from
    /// `<data_dir>/<round_id>/analysis/ground_truth_seed_index=<seed_index>.json`.
    pub fn load(round_id: &str, seed_index: usize) -> Result<Self> {
        let path = data_dir()
            .join(round_id)
            .join("analysis")
            .join(format!("ground_truth_seed_index={}.json", seed_index));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading ground truth from {:?}", path))?;
        let data: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing ground truth JSON from {:?}", path))?;
        Ok(data)
    }
}

// ── Round details ─────────────────────────────────────────────────────────────

/// One seed's initial map + settlement layout inside a round.
#[derive(Debug, Deserialize)]
pub struct InitialState {
    pub grid: Grid,
    pub settlements: Vec<InitialSettlement>,
}

/// Full round metadata as stored in `details.json`.
#[derive(Debug, Deserialize)]
pub struct RoundDetails {
    pub id: String,
    pub round_number: u32,
    pub map_width: usize,
    pub map_height: usize,
    pub seeds_count: usize,
    pub initial_states: Vec<InitialState>,
}

impl RoundDetails {
    /// Load details from `<data_dir>/<round_id>/details.json`.
    pub fn load(round_id: &str) -> Result<Self> {
        let path = data_dir().join(round_id).join("details.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading round details from {:?}", path))?;
        let data: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing round details JSON from {:?}", path))?;
        Ok(data)
    }
}

// ── Query result ──────────────────────────────────────────────────────────────

/// The spatial viewport that was queried.
#[derive(Debug, Deserialize)]
pub struct QueryViewport {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// A single query result as stored in the `query/` subdirectory.
#[derive(Debug, Deserialize)]
pub struct QueryResult {
    pub grid: Grid,
    pub settlements: Vec<ReplaySettlement>,
    pub viewport: QueryViewport,
    pub width: usize,
    pub height: usize,
    pub queries_used: usize,
    pub queries_max: usize,
}

// ── Round discovery ───────────────────────────────────────────────────────────

/// Return the IDs of all rounds that have at least
/// `analysis/replay_seed_index=0.json` present.
pub fn list_rounds_with_analysis() -> Result<Vec<String>> {
    let base = data_dir();
    let mut rounds = Vec::new();

    let entries = std::fs::read_dir(&base)
        .with_context(|| format!("reading data directory {:?}", base))?;

    for entry in entries {
        let entry = entry.with_context(|| format!("iterating data directory {:?}", base))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let sentinel = path.join("analysis").join("replay_seed_index=0.json");
        if sentinel.exists() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                rounds.push(name.to_owned());
            }
        }
    }

    rounds.sort();
    Ok(rounds)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ROUND_ID: &str = "ae78003a-4efe-425a-881a-d16a39bca0ad";

    #[test]
    fn test_data_dir_exists() {
        let dir = data_dir();
        assert!(dir.is_dir(), "data_dir() returned {:?} which is not a directory", dir);
    }

    #[test]
    fn test_load_replay() {
        let replay = ReplayData::load(ROUND_ID, 0).expect("load replay");
        assert_eq!(replay.round_id, ROUND_ID);
        assert_eq!(replay.seed_index, 0);
        assert!(replay.width > 0);
        assert!(replay.height > 0);
        assert!(!replay.frames.is_empty());
        // Grid dimensions must match declared width/height.
        let f0 = &replay.frames[0];
        assert_eq!(f0.grid.len(), replay.height);
        assert_eq!(f0.grid[0].len(), replay.width);
    }

    #[test]
    fn test_load_ground_truth() {
        let gt = GroundTruthData::load(ROUND_ID, 0).expect("load ground truth");
        assert_eq!(gt.width, 40);
        assert_eq!(gt.height, 40);
        assert_eq!(gt.ground_truth.len(), gt.height);
        assert_eq!(gt.ground_truth[0].len(), gt.width);
        assert_eq!(gt.initial_grid.len(), gt.height);
    }

    #[test]
    fn test_load_round_details() {
        let details = RoundDetails::load(ROUND_ID).expect("load details");
        assert_eq!(details.id, ROUND_ID);
        assert_eq!(details.round_number, 6);
        assert_eq!(details.map_width, 40);
        assert_eq!(details.map_height, 40);
        assert!(!details.initial_states.is_empty());
        let s0 = &details.initial_states[0];
        assert_eq!(s0.grid.len(), details.map_height);
        assert_eq!(s0.grid[0].len(), details.map_width);
    }

    #[test]
    fn test_list_rounds_with_analysis() {
        let rounds = list_rounds_with_analysis().expect("list rounds");
        assert!(!rounds.is_empty(), "expected at least one round with analysis");
        assert!(
            rounds.contains(&ROUND_ID.to_owned()),
            "expected {} in round list",
            ROUND_ID
        );
    }

    #[test]
    fn test_distribution_grid_shape() {
        let gt = GroundTruthData::load(ROUND_ID, 0).expect("load ground truth");
        // Each cell is a 6-element array.
        let cell = gt.ground_truth[0][0];
        assert_eq!(cell.len(), 6);
        // Probabilities sum to ~1.
        let sum: f64 = cell.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities sum to {}", sum);
    }
}
