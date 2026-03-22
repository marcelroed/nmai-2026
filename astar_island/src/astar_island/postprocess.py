"""Post-process Monte Carlo predictions to fix systematic biases.

Takes raw MC predictions + initial grid and applies:
1. Static terrain: hard copy ocean/mountain
2. Settlement: spatial smoothing (Gaussian blur) preserving total mass
3. Ruin: derived from settlement probability (negative correlation)
4. Forest: reduce near settlements
5. Port: hard ocean-adjacency constraint
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


# Terrain codes from initial grid
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5

# Prediction class indices
C_EMPTY = 0
C_SETTLEMENT = 1
C_PORT = 2
C_RUIN = 3
C_FOREST = 4
C_MOUNTAIN = 5


def compute_port_fraction(round_id: str, data_dir: str | Path = "data") -> float:
    """Compute port_fraction from replay data at step 50 for a specific round.

    At step 50, finds all ocean-adjacent cells that are SETTLEMENT or PORT,
    and returns the fraction that are PORT. Averaged across all replays.
    """
    data_dir = Path(data_dir)
    fractions: list[float] = []

    for replay_path in sorted(data_dir.glob(f"{round_id}/analysis/replay_seed_index=*.json")):
        replay = json.loads(replay_path.read_text())
        frames = replay["frames"]
        if len(frames) <= 50:
            continue
        grid = np.array(frames[50]["grid"])
        ocean_adjacent = _compute_ocean_adjacency(grid)
        coastal_settled = ocean_adjacent & ((grid == SETTLEMENT) | (grid == PORT))
        n_coastal = coastal_settled.sum()
        if n_coastal == 0:
            continue
        n_ports = (ocean_adjacent & (grid == PORT)).sum()
        fractions.append(n_ports / n_coastal)

    if not fractions:
        return 0.4  # fallback if no replay data available
    return float(np.mean(fractions))


def compute_ruin_fraction(round_id: str, data_dir: str | Path = "data") -> float:
    """Compute ruin_fraction from replay data (steps 40-50) for a specific round.

    For each step t in [40, 50), compute the fraction of settlement cells
    at step t that become ruin cells at step t+1. Returns the average
    across all steps and all available replays for the given round.
    """
    data_dir = Path(data_dir)
    fractions: list[float] = []

    for replay_path in sorted(data_dir.glob(f"{round_id}/analysis/replay_seed_index=*.json")):
        replay = json.loads(replay_path.read_text())
        frames = replay["frames"]
        if len(frames) <= 50:
            continue
        for t in range(40, 50):
            grid_t = np.array(frames[t]["grid"])
            grid_t1 = np.array(frames[t + 1]["grid"])
            settlements_t = grid_t == SETTLEMENT
            n_settlements = settlements_t.sum()
            if n_settlements == 0:
                continue
            became_ruin = settlements_t & (grid_t1 == RUIN)
            fractions.append(became_ruin.sum() / n_settlements)

    if not fractions:
        return 0.15  # fallback if no replay data available
    return float(np.mean(fractions))


def project_to_feasible(
    prediction: np.ndarray,
    initial_grid: np.ndarray,
) -> np.ndarray:
    """Project a (H, W, 6) probability matrix onto the feasible set.

    Feasibility constraints:
    - Ocean cells: hard [1, 0, 0, 0, 0, 0]
    - Mountain cells: hard [0, 0, 0, 0, 0, 1]
    - Dynamic cells: mountain class = 0, port = 0 if not ocean-adjacent,
      all probs >= 0, sum to 1
    """
    pred = prediction.copy()

    ocean_mask = initial_grid == OCEAN
    mountain_mask = initial_grid == MOUNTAIN
    dynamic = ~ocean_mask & ~mountain_mask

    pred[ocean_mask] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pred[mountain_mask] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    ocean_adjacent = _compute_ocean_adjacency(initial_grid)
    pred[~ocean_adjacent & dynamic, C_PORT] = 0.0

    dynamic_cells = pred[dynamic]
    dynamic_cells[:, C_MOUNTAIN] = 0.0
    np.maximum(dynamic_cells, 0.0, out=dynamic_cells)

    row_sums = dynamic_cells.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) == 0
    dynamic_cells[zero_rows, :5] = 0.2
    row_sums[zero_rows] = 1.0

    dynamic_cells /= row_sums
    pred[dynamic] = dynamic_cells

    return pred



def postprocess(
    prediction: np.ndarray,
    initial_grid: np.ndarray,
    ruin_fraction: float,
    *args,
    **kwargs: dict,
) -> np.ndarray:
    """Naive postprocessing: set ruins = ruin_fraction * settlements, then project to feasible."""
    pred = prediction.copy()
    pred[:, :, C_RUIN] = pred[:, :, C_SETTLEMENT] * ruin_fraction
    return project_to_feasible(pred, initial_grid)



# def postprocess(
#     prediction: np.ndarray,
#     initial_grid: np.ndarray,
#     *,
#     settlement_sigma: float = 0.0,
#     ruin_fraction: float | None = None,
#     port_fraction: float | None = None,
#     forest_settlement_decay: float = 0.1,
#     prob_floor: float = 0.01,
# ) -> np.ndarray:
#     """Apply all post-processing corrections to a raw MC prediction.

#     Args:
#         prediction: (H, W, 6) probability tensor from Monte Carlo
#         initial_grid: (H, W) terrain codes from initial state
#         settlement_sigma: Gaussian blur sigma for settlement smoothing
#         ruin_fraction: fraction of settlement probability to allocate to ruin.
#             If None, computed from replay data via compute_ruin_fraction().
#         port_fraction: fraction of coastal settlements that are ports (from replay data).
#             Used as the port boost factor for coastal cells.
#             If None, defaults to 0.4.
#         forest_settlement_decay: how much to reduce forest near settlements
#         prob_floor: minimum probability for any class on dynamic cells

#     Returns:
#         (H, W, 6) corrected probability tensor
#     """
#     if ruin_fraction is None:
#         ruin_fraction = compute_ruin_fraction()
#     if port_fraction is None:
#         port_fraction = 0.4

#     pred = prediction.copy()
#     h, w = pred.shape[:2]

#     # === 1. Static terrain: hard copy ===
#     ocean_mask = (initial_grid == OCEAN)
#     mountain_mask = (initial_grid == MOUNTAIN)
#     dynamic = ~ocean_mask & ~mountain_mask

#     pred = project_to_feasible(pred, initial_grid)

#     # === 2. Settlement: spatial smoothing ===
#     # Smooth settlement probability with Gaussian blur, preserving total mass
#     settlement_raw = pred[:, :, C_SETTLEMENT].copy()
#     settlement_total = settlement_raw[dynamic].sum()

#     # Only smooth dynamic cells (set static to 0 before blur, restore after)
#     settlement_for_blur = settlement_raw.copy()
#     settlement_for_blur[~dynamic] = 0.0
#     settlement_smooth = gaussian_filter(settlement_for_blur, sigma=settlement_sigma)
#     settlement_smooth[~dynamic] = 0.0

#     # Renormalize to preserve total settlement mass
#     smooth_total = settlement_smooth[dynamic].sum()
#     if smooth_total > 0:
#         settlement_smooth[dynamic] *= settlement_total / smooth_total

#     pred[:, :, C_SETTLEMENT] = settlement_smooth

#     # === 3. Ruin: derive from settlement probability ===
#     # Ruin probability is settlement probability * ruin_fraction
#     # This is a much less noisy estimate of the ruin probability than through monte carlo simulation.
#     ruin_from_settlement = settlement_smooth * ruin_fraction
#     ruin_from_settlement[~dynamic] = 0.0
#     pred[:, :, C_RUIN] = ruin_from_settlement

#     # === 4. Forest: reduce near settlements ===
#     # Settlements clear forests. Reduce forest probability where settlement
#     # probability is high.
#     forest_reduction = settlement_smooth * forest_settlement_decay
#     pred[:, :, C_FOREST] = np.maximum(
#         pred[:, :, C_FOREST] - forest_reduction, 0.0
#     )
#     pred[~dynamic, C_FOREST] = 0.0
#     pred[:, :, C_FOREST] = np.maximum(pred[:, :, C_FOREST], settlement_smooth * forest_settlement_decay)

#     # === 5. Port: hard ocean-adjacency constraint ===
#     # Ports can only appear adjacent to ocean
#     ocean_adjacent = _compute_ocean_adjacency(initial_grid)

#     # Both (settlement+port) * port_fraction and port are estimates for the port probability. We use a weighted average of the two.
#     port_pred = pred[:, :, C_PORT]
#     port_boost_map = (settlement_smooth + port_pred) * port_fraction
#     pred[:, :, C_PORT] = 0.4 * port_pred + 0.6 * port_boost_map

#     # Zero out port probability for non-coastal cells
#     pred[~ocean_adjacent, C_PORT] = 0.0
#     pred[~dynamic, C_PORT] = 0.0

#     # === 6. Renormalize ===
#     # Apply floor on dynamic cells, then renormalize
#     for y in range(h):
#         for x in range(w):
#             if not dynamic[y, x]:
#                 continue
#             cell = pred[y, x]
#             # Apply floor
#             cell = np.maximum(cell, prob_floor)
#             # make the mountain cell probability 0
#             cell[C_MOUNTAIN] = 0.0
#             # Renormalize
#             cell /= cell.sum()
#             pred[y, x] = cell

#     return pred


def _compute_ocean_adjacency(grid: np.ndarray) -> np.ndarray:
    """Compute a boolean mask of cells adjacent to ocean (4-connected, cardinal only)."""
    h, w = grid.shape
    ocean = (grid == OCEAN)
    adjacent = np.zeros_like(ocean)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(ocean, dy, axis=0), dx, axis=1)
        # Zero out wrapped edges
        if dy == -1:
            shifted[-1, :] = False
        elif dy == 1:
            shifted[0, :] = False
        if dx == -1:
            shifted[:, -1] = False
        elif dx == 1:
            shifted[:, 0] = False
        adjacent |= shifted
    return adjacent


def postprocess_predictions(
    predictions: dict[int, list],
    initial_states: list[dict],
    round_id: str,
) -> dict[int, list]:
    """Post-process all seed predictions.

    Args:
        predictions: {seed_idx: H×W×6 list}
        initial_states: list of initial state dicts with 'grid' key
        round_id: round to compute ruin fraction from

    Returns:
        {seed_idx: H×W×6 corrected list}
    """
    ruin_fraction = compute_ruin_fraction(round_id)
    print(f"Ruin fraction: {ruin_fraction}")
    result = {}
    for seed_idx, pred_list in predictions.items():
        pred = np.array(pred_list)
        grid = np.array(initial_states[seed_idx]["grid"])
        corrected = postprocess(pred, grid, ruin_fraction)
        result[seed_idx] = corrected.tolist()
    return result
