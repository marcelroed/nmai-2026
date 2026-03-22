use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, FixedOffset, NaiveDate};
use serde::Deserialize;

use crate::data::{Grid, InitialState, TerrainType, details_path, query_dir};
pub type ObservedMask = Vec<Vec<bool>>;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct LoadedRoundDetails {
    pub(super) id: String,
    pub(super) round_number: u32,
    pub(super) event_date: NaiveDate,
    pub(super) status: String,
    #[serde(rename = "map_width")]
    pub(super) n_cols: usize,
    #[serde(rename = "map_height")]
    pub(super) n_rows: usize,
    pub(super) prediction_window_minutes: u32,
    pub(super) started_at: DateTime<FixedOffset>,
    pub(super) closes_at: DateTime<FixedOffset>,
    pub(super) round_weight: f64,
    #[serde(rename = "seeds_count")]
    pub(super) map_count: usize,
    pub(super) initial_states: Vec<InitialState>,
}

#[derive(Clone, Debug)]
pub struct QueryRecord {
    pub map_idx: usize,
    pub query_index: usize,
    pub result: QueryResult,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QuerySettlement {
    #[serde(rename = "y")]
    pub row: usize,
    #[serde(rename = "x")]
    pub col: usize,
    pub has_port: bool,
    pub alive: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QueryViewport {
    #[serde(rename = "y")]
    pub row: usize,
    #[serde(rename = "x")]
    pub col: usize,
    #[serde(rename = "h")]
    pub n_rows: usize,
    #[serde(rename = "w")]
    pub n_cols: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QueryResult {
    pub grid: Grid,
    pub settlements: Vec<QuerySettlement>,
    pub viewport: QueryViewport,
    pub queries_used: Option<u32>,
    pub queries_max: Option<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct TerrainCounts {
    counts: BTreeMap<TerrainType, usize>,
}

impl TerrainCounts {
    pub fn add(&mut self, terrain: TerrainType) {
        *self.counts.entry(terrain).or_insert(0) += 1;
    }

    pub fn get(&self, terrain: TerrainType) -> usize {
        self.counts.get(&terrain).copied().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct MapView {
    pub map_idx: usize,
    pub base_grid: Grid,
    pub stitched_grid: Grid,
    pub observed_mask: ObservedMask,
    pub query_count: usize,
    pub settlement_count: usize,
    pub alive_settlement_count: usize,
    pub port_count: usize,
    pub counts: TerrainCounts,
}

#[derive(Clone, Debug)]
pub struct LoadedRound {
    pub details_path: PathBuf,
    pub query_dir: PathBuf,
    pub details: LoadedRoundDetails,
    pub queries: Vec<QueryRecord>,
    pub map_views: Vec<MapView>,
}

#[derive(Clone, Debug)]
pub struct CachedRoundSummary {
    pub id: String,
    pub round_number: u32,
    pub event_date: NaiveDate,
}

impl LoadedRound {
    pub fn from_round_id(round_id: &str) -> Result<Self> {
        let details_path = details_path(round_id);
        let query_dir = query_dir(round_id);
        let details = load_details_from_path(&details_path)?;
        Self::from_parts(details_path, query_dir, details)
    }

    pub fn from_details_path(details_path: impl Into<PathBuf>) -> Result<Self> {
        let details_path = details_path.into();
        let query_dir = details_path
            .parent()
            .map(|parent| parent.join("query"))
            .ok_or_else(|| anyhow!("details path has no parent: {}", details_path.display()))?;
        Self::from_paths(details_path, query_dir)
    }

    pub fn from_paths(
        details_path: impl Into<PathBuf>,
        query_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        let details_path = details_path.into();
        let query_dir = query_dir.into();
        let details = load_details_from_path(&details_path)?;
        Self::from_parts(details_path, query_dir, details)
    }

    fn from_parts(
        details_path: PathBuf,
        query_dir: PathBuf,
        details: LoadedRoundDetails,
    ) -> Result<Self> {
        let queries = load_queries(&query_dir)?;
        let map_views = build_map_views(&details, &queries)?;
        Ok(Self {
            details_path,
            query_dir,
            details,
            queries,
            map_views,
        })
    }
}

pub fn list_cached_rounds(data_dir: &Path) -> Result<Vec<CachedRoundSummary>> {
    if !data_dir.exists() {
        return Ok(Vec::new());
    }

    let mut rounds = Vec::new();
    for entry in fs::read_dir(data_dir)
        .with_context(|| format!("failed to read {}", data_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let details_path = path.join("details.json");
        if !details_path.exists() {
            continue;
        }

        let details = load_details_from_path(&details_path)?;
        rounds.push(CachedRoundSummary {
            id: details.id,
            round_number: details.round_number,
            event_date: details.event_date,
        });
    }

    rounds.sort_by_key(|round| (round.round_number, round.event_date, round.id.clone()));
    Ok(rounds)
}

fn build_map_views(details: &LoadedRoundDetails, queries: &[QueryRecord]) -> Result<Vec<MapView>> {
    let mut queries_by_map: HashMap<usize, Vec<&QueryRecord>> = HashMap::new();
    for query in queries {
        queries_by_map.entry(query.map_idx).or_default().push(query);
    }

    let mut map_views = Vec::with_capacity(details.initial_states.len());
    for (map_idx, initial_state) in details.initial_states.iter().enumerate() {
        map_views.push(build_map_view(map_idx, initial_state, details, &mut queries_by_map)?);
    }

    Ok(map_views)
}

fn build_map_view(
    map_idx: usize,
    initial_state: &InitialState,
    details: &LoadedRoundDetails,
    queries_by_map: &mut HashMap<usize, Vec<&QueryRecord>>,
) -> Result<MapView> {
    let mut stitched_grid = initial_state.grid.clone();
    let mut observed_mask = vec![vec![false; details.n_cols]; details.n_rows];
    let mut settlements = initial_state
        .settlements
        .iter()
        .map(|settlement| {
            (
                (settlement.row, settlement.col),
                (settlement.alive, settlement.has_port),
            )
        })
        .collect::<HashMap<_, _>>();

    let mut query_count = 0usize;
    if let Some(map_queries) = queries_by_map.get_mut(&map_idx) {
        map_queries.sort_by_key(|query| {
            (
                query.result.viewport.row,
                query.result.viewport.col,
                query.query_index,
            )
        });

        for query in map_queries {
            query_count += 1;
            overlay_query(&mut stitched_grid, &mut observed_mask, &mut settlements, query)?;
        }
    }

    let counts = count_grid(&stitched_grid);
    let settlement_count = counts.get(TerrainType::Settlement) + counts.get(TerrainType::Port);
    let port_count = counts.get(TerrainType::Port);
    let alive_settlement_count = settlements
        .iter()
        .filter(|((row, col), (alive, _))| {
            *alive
                && matches!(
                    stitched_grid[*row][*col],
                    TerrainType::Settlement | TerrainType::Port
                )
        })
        .count();

    Ok(MapView {
        map_idx,
        base_grid: initial_state.grid.clone(),
        stitched_grid,
        observed_mask,
        query_count,
        settlement_count,
        alive_settlement_count,
        port_count,
        counts,
    })
}

fn overlay_query(
    stitched_grid: &mut Grid,
    observed_mask: &mut ObservedMask,
    settlements: &mut HashMap<(usize, usize), (bool, bool)>,
    query: &QueryRecord,
) -> Result<()> {
    let top = query.result.viewport.row;
    let left = query.result.viewport.col;
    if query.result.grid.len() != query.result.viewport.n_rows {
        bail!(
            "query height mismatch for map {}: grid={}, viewport={}",
            query.map_idx,
            query.result.grid.len(),
            query.result.viewport.n_rows
        );
    }
    if query
        .result
        .grid
        .iter()
        .any(|row| row.len() != query.result.viewport.n_cols)
    {
        bail!(
            "query width mismatch for map {}: expected {}",
            query.map_idx,
            query.result.viewport.n_cols
        );
    }

    for (r_off, row) in query.result.grid.iter().enumerate() {
        for (c_off, cell) in row.iter().enumerate() {
            let row_idx = top + r_off;
            let col_idx = left + c_off;

            let grid_row = stitched_grid.get_mut(row_idx).ok_or_else(|| {
                anyhow!(
                    "query row {} out of bounds for map {}",
                    row_idx,
                    query.map_idx
                )
            })?;
            let observed_row = observed_mask.get_mut(row_idx).ok_or_else(|| {
                anyhow!(
                    "query mask row {} out of bounds for map {}",
                    row_idx,
                    query.map_idx
                )
            })?;

            if col_idx >= grid_row.len() || col_idx >= observed_row.len() {
                bail!("query col {} out of bounds for map {}", col_idx, query.map_idx);
            }

            grid_row[col_idx] = *cell;
            observed_row[col_idx] = true;
        }
    }

    for settlement in &query.result.settlements {
        settlements.insert(
            (settlement.row, settlement.col),
            (settlement.alive, settlement.has_port),
        );
    }

    Ok(())
}

fn count_grid(grid: &Grid) -> TerrainCounts {
    let mut counts = TerrainCounts::default();
    for row in grid {
        for cell in row {
            counts.add(*cell);
        }
    }
    counts
}

fn load_queries(query_dir: &Path) -> Result<Vec<QueryRecord>> {
    if !query_dir.exists() {
        return Ok(Vec::new());
    }

    let mut queries = Vec::new();
    for entry in fs::read_dir(query_dir)
        .with_context(|| format!("failed to read {}", query_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow!("invalid query filename: {}", path.display()))?;
        let parsed = ParsedQueryName::parse(file_name)?;
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read query file at {}", path.display()))?;
        let result: QueryResult = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse json in {}", path.display()))?;

        queries.push(QueryRecord {
            map_idx: parsed.map_idx,
            query_index: parsed.query_index,
            result,
        });
    }

    Ok(queries)
}

#[derive(Debug)]
struct ParsedQueryName {
    map_idx: usize,
    query_index: usize,
}

impl ParsedQueryName {
    fn parse(file_name: &str) -> Result<Self> {
        let stem = file_name
            .strip_suffix(".json")
            .ok_or_else(|| anyhow!("query filename is not json: {file_name}"))?;

        let map_idx = parse_usize_field(stem, "map_idx=")?;
        let query_index = match parse_usize_field(stem, "run_seed_idx=") {
            Ok(value) => value,
            Err(_) => parse_usize_field(stem, "snapshot_seed=")?,
        };

        Ok(Self {
            map_idx,
            query_index,
        })
    }
}

fn parse_usize_field(input: &str, key: &str) -> Result<usize> {
    let start = input
        .find(key)
        .ok_or_else(|| anyhow!("missing {key} in {input}"))?
        + key.len();
    let digits = input[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();

    if digits.is_empty() {
        bail!("missing numeric value for {key} in {input}");
    }

    digits
        .parse::<usize>()
        .with_context(|| format!("failed to parse {key}{digits}"))
}

fn load_details_from_path(path: &Path) -> Result<LoadedRoundDetails> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read details file at {}", path.display()))?;
    let details = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse json in {}", path.display()))?;
    Ok(details)
}
