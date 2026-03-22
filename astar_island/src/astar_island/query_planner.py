"""Strategic viewport placement: tile every seed for full spatial coverage.

Design principles:
- Each query runs a DIFFERENT stochastic simulation (different sim_seed)
- Hidden parameters are shared across seeds, so observing all seeds helps
  inference, and each seed needs its own predictions
- Tile every seed with the 9-position grid for full 40x40 coverage
- Extra budget beyond full tiling goes to repeats (round-robin across seeds)

Strategy (50 queries, 5 seeds → 10 per seed):
  Each seed: 9 unique tile positions + 1 repeat = 10 queries
  → Full spatial coverage on all 5 seeds
"""


# 9 viewport positions that tile the 40x40 map with 15x15 windows
TILE_POSITIONS_9 = [
    (0, 0),   (0, 13),   (0, 25),
    (13, 0),  (13, 13),  (13, 25),
    (25, 0),  (25, 13),  (25, 25),
]


def plan_queries(initial_states: list[dict], budget: int = 50) -> list[dict]:
    """Plan viewport queries: tile every seed, then distribute extra as repeats.

    Returns list of {seed_index, viewport_x, viewport_y, viewport_w, viewport_h}.
    """
    n_seeds = len(initial_states) if initial_states else 5
    n_tiles = len(TILE_POSITIONS_9)
    per_seed = budget // n_seeds
    leftover = budget - per_seed * n_seeds

    queries: list[dict] = []
    for seed_idx in range(n_seeds):
        n_queries = per_seed + (1 if seed_idx < leftover else 0)
        for i in range(n_queries):
            vx, vy = TILE_POSITIONS_9[i % n_tiles]
            queries.append({
                "seed_index": seed_idx,
                "viewport_x": vx,
                "viewport_y": vy,
                "viewport_w": min(15, 40 - vx),
                "viewport_h": min(15, 40 - vy),
            })

    return queries
