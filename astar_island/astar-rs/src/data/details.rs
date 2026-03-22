use std::fmt;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::{Grid, Settlement, details_path, read_json_file};

#[derive(Clone, Debug, Default, Deserialize)]
pub struct InitialState {
    pub grid: Grid,
    pub settlements: Vec<Settlement>,
}

#[derive(Clone, Deserialize)]
pub struct RoundDetails {
    pub id: String,
    pub round_number: u32,
    #[serde(skip)]
    pub map_index: usize,
    #[serde(skip)]
    pub initial_state: InitialState,
    #[serde(rename = "initial_states")]
    raw_initial_states: Vec<InitialState>,
}

impl RoundDetails {
    pub fn from_round_id_and_map(round_id: &str, map_index: usize) -> Result<Self> {
        let mut details: Self = read_json_file(&details_path(round_id))?;
        details.initial_state = details
            .raw_initial_states
            .get(map_index)
            .cloned()
            .with_context(|| {
                format!(
                    "map index {map_index} out of bounds for round {} with {} maps",
                    details.id,
                    details.raw_initial_states.len()
                )
            })?;
        details.map_index = map_index;
        details.raw_initial_states.clear();
        Ok(details)
    }
}

impl fmt::Debug for RoundDetails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = crate::visualize::round_details_debug_snapshot(self)
            .expect("RoundDetails debug snapshot should always render");
        f.write_str(&snapshot)
    }
}
