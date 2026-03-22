use anyhow::Result;
use serde::Deserialize;

use super::{DistributionGrid, Grid, ground_truth_path, read_json_file};

#[derive(Debug, Deserialize)]
pub struct GroundTruthData {
    pub width: usize,
    pub height: usize,
    pub initial_grid: Grid,
    pub ground_truth: DistributionGrid,
    pub prediction: Option<DistributionGrid>,
    pub score: Option<f64>,
}

impl GroundTruthData {
    pub fn from_round_id_and_map(round_id: &str, map_index: usize) -> Result<Self> {
        read_json_file(&ground_truth_path(round_id, map_index))
    }
}
