use std::path::{Path, PathBuf};

pub const DEFAULT_DATA_DIR: &str = "../data";

pub fn round_data_dir(round_id: &str) -> PathBuf {
    Path::new(DEFAULT_DATA_DIR).join(round_id)
}

pub fn details_path(round_id: &str) -> PathBuf {
    round_data_dir(round_id).join("details.json")
}

pub fn query_dir(round_id: &str) -> PathBuf {
    round_data_dir(round_id).join("query")
}

pub fn analysis_dir(round_id: &str) -> PathBuf {
    round_data_dir(round_id).join("analysis")
}

pub fn ground_truth_path(round_id: &str, map_index: usize) -> PathBuf {
    analysis_dir(round_id).join(format!("ground_truth_seed_index={map_index}.json"))
}

pub fn replay_path(round_id: &str, map_index: usize) -> PathBuf {
    analysis_dir(round_id).join(format!("replay_seed_index={map_index}.json"))
}
