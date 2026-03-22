use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::Deserialize;

mod details;
mod ground_truth;
mod paths;
mod replay;
mod terrain;

pub use details::{InitialState, RoundDetails};
pub use ground_truth::GroundTruthData;
pub use paths::{
    DEFAULT_DATA_DIR, analysis_dir, details_path, ground_truth_path, query_dir, replay_path,
    round_data_dir,
};
pub use replay::{ReplayData, ReplayFrame, ReplaySettlement, SettlementMetrics};
pub use terrain::TerrainType;

pub type Grid = Vec<Vec<TerrainType>>;
pub type CellDistribution = [f64; 6];
pub type DistributionGrid = Vec<Vec<CellDistribution>>;

#[derive(Clone, Debug, Deserialize)]
pub struct Settlement {
    #[serde(rename = "y")]
    pub row: usize,
    #[serde(rename = "x")]
    pub col: usize,
    pub has_port: bool,
    pub alive: bool,
}

pub fn read_json_file<T>(path: &Path) -> Result<T>
where
    T: DeserializeOwned,
{
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read json file at {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse json in {}", path.display()))
}
