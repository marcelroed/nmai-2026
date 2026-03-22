use anyhow::Result;
use serde::Deserialize;

use super::{Grid, Settlement, read_json_file, replay_path};

#[derive(Debug, Deserialize)]
pub struct SettlementMetrics {
    pub population: f64,
    pub food: f64,
    pub wealth: f64,
    pub defense: f64,
    pub owner_id: usize,
}

#[derive(Debug, Deserialize)]
pub struct ReplaySettlement {
    #[serde(flatten)]
    pub state: Settlement,
    #[serde(flatten)]
    pub metrics: SettlementMetrics,
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
    #[serde(rename = "seed_index")]
    pub map_index: usize,
    pub sim_seed: u64,
    pub width: usize,
    pub height: usize,
    pub frames: Vec<ReplayFrame>,
}

impl ReplayData {
    pub fn from_round_id_and_map(round_id: &str, map_index: usize) -> Result<Self> {
        read_json_file(&replay_path(round_id, map_index))
    }
}
