"""
Food analysis v6: Try wider terrain radius, per-cell features, and check
what terrain codes actually appear in the data.

Reproduce:  python3 data-analysis/analyze_food_v6.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


def terrain_counts(grid, x, y, radius):
    """Count terrain types within Chebyshev distance `radius`, excluding self."""
    h, w = len(grid), len(grid[0])
    counts = defaultdict(int)
    for ny in range(max(0, y - radius), min(h, y + radius + 1)):
        for nx in range(max(0, x - radius), min(w, x + radius + 1)):
            if nx == x and ny == y:
                continue
            counts[grid[ny][nx]] += 1
    return counts


def load_replays_by_round():
    by_round: dict[str, list] = defaultdict(list)
    for rd in sorted(DATA_DIR.iterdir()):
        if not rd.is_dir():
            continue
        ad = rd / "analysis"
        if not ad.exists():
            continue
        for f in sorted(ad.glob("replay_seed_index=*.json")):
            with open(f) as fh:
                data = json.load(fh)
            by_round[data["round_id"]].append(data)
    return dict(by_round)


def fit_ols(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    res = y - pred
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def main():
    rounds = load_replays_by_round()
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays\n")

    # Check terrain codes in data
    sample_replay = list(rounds.values())[0][0]
    grid = sample_replay["frames"][0]["grid"]
    all_codes = set()
    for row in grid:
        all_codes.update(row)
    print(f"Terrain codes in data: {sorted(all_codes)}")
    print(f"Grid size: {len(grid[0])}×{len(grid)}\n")

    for rid in sorted(rounds):
        replays = rounds[rid]
        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}")
        print(f"{'=' * 72}")

        # Step 0→1 only
        rows = []
        for replay in replays:
            frames = replay["frames"]
            if len(frames) < 2:
                continue
            fb, fa = frames[0], frames[1]
            grid = fb["grid"]
            before_map = {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}
            after_map = {}
            for s in fa["settlements"]:
                if s["alive"]:
                    k = (s["x"], s["y"])
                    if k not in after_map:
                        after_map[k] = s

            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                # Radius 1 counts
                c1 = terrain_counts(grid, x, y, 1)
                # Radius 2 counts (ring only: r2 minus r1)
                c2_full = terrain_counts(grid, x, y, 2)
                c2_ring = {k: c2_full.get(k, 0) - c1.get(k, 0) for k in set(c1) | set(c2_full)}
                # Radius 3 ring
                c3_full = terrain_counts(grid, x, y, 3)
                c3_ring = {k: c3_full.get(k, 0) - c2_full.get(k, 0) for k in set(c2_full) | set(c3_full)}

                # Features: separate plains vs empty at r=1
                rows.append((
                    sb["population"], sb["food"],
                    sa["food"],
                    # Radius 1: separate each type
                    c1.get(PLAINS, 0), c1.get(EMPTY, 0), c1.get(FOREST, 0),
                    c1.get(MOUNTAIN, 0), c1.get(OCEAN, 0),
                    c1.get(SETTLEMENT, 0) + c1.get(PORT, 0) + c1.get(RUIN, 0),
                    # Radius 2 ring (grouped)
                    c2_ring.get(PLAINS, 0) + c2_ring.get(EMPTY, 0),
                    c2_ring.get(FOREST, 0),
                    c2_ring.get(MOUNTAIN, 0),
                    c2_ring.get(OCEAN, 0),
                    # Radius 3 ring (grouped)
                    c3_ring.get(PLAINS, 0) + c3_ring.get(EMPTY, 0),
                    c3_ring.get(FOREST, 0),
                    c3_ring.get(MOUNTAIN, 0),
                    c3_ring.get(OCEAN, 0),
                ))

        arr = np.array(rows, dtype=np.float64)
        food_delta = arr[:, 2] - arr[:, 1]
        n = len(arr)

        # ── Model 1: radius 1 (plains+empty combined) ─────────────────────
        X1 = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0],  # intercept, food, pop
            arr[:, 3] + arr[:, 4],  # plains+empty r1
            arr[:, 5],              # forest r1
        ])
        c1, r2_1, mae1, _ = fit_ols(X1, food_delta)
        print(f"  R1 combined (int+food+pop+pl_em+fo): R²={r2_1:.6f}")

        # ── Model 1b: radius 1 (plains separate from empty) ───────────────
        X1b = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0],
            arr[:, 3],  # plains r1
            arr[:, 4],  # empty r1
            arr[:, 5],  # forest r1
        ])
        c1b, r2_1b, mae1b, _ = fit_ols(X1b, food_delta)
        print(f"  R1 separate (int+food+pop+pl+empty+fo): R²={r2_1b:.6f}")
        print(f"    plains={c1b[3]:.6f}, empty={c1b[4]:.6f}, forest={c1b[5]:.6f}")

        # ── Model 2: radius 1 + ocean ─────────────────────────────────────
        X2 = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0],
            arr[:, 3] + arr[:, 4], arr[:, 5], arr[:, 7],  # pl+em, fo, ocean
        ])
        c2, r2_2, mae2, _ = fit_ols(X2, food_delta)
        print(f"  R1 + ocean: R²={r2_2:.6f}  [ocean={c2[5]:.6f}]")

        # ── Model 3: radius 1 all types ───────────────────────────────────
        X3 = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0],
            arr[:, 3] + arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8],
        ])
        c3, r2_3, mae3, _ = fit_ols(X3, food_delta)
        print(f"  R1 all types: R²={r2_3:.6f}  "
              f"[pl+em={c3[3]:.6f}, fo={c3[4]:.6f}, mt={c3[5]:.6f}, oc={c3[6]:.6f}, st={c3[7]:.6f}]")

        # ── Model 4: + radius 2 ring ──────────────────────────────────────
        X4 = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0],
            arr[:, 3] + arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7],
            arr[:, 9], arr[:, 10], arr[:, 11], arr[:, 12],  # r2 ring
        ])
        c4, r2_4, mae4, _ = fit_ols(X4, food_delta)
        print(f"  R1+R2 ring: R²={r2_4:.6f}")
        print(f"    R2 ring: pl+em={c4[7]:.6f}, fo={c4[8]:.6f}, mt={c4[9]:.6f}, oc={c4[10]:.6f}")

        # ── Model 5: + radius 3 ring ──────────────────────────────────────
        X5 = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0],
            arr[:, 3] + arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7],
            arr[:, 9], arr[:, 10], arr[:, 11], arr[:, 12],
            arr[:, 13], arr[:, 14], arr[:, 15], arr[:, 16],  # r3 ring
        ])
        c5, r2_5, mae5, _ = fit_ols(X5, food_delta)
        print(f"  R1+R2+R3: R²={r2_5:.6f}")
        print(f"    R3 ring: pl+em={c5[11]:.6f}, fo={c5[12]:.6f}, mt={c5[13]:.6f}, oc={c5[14]:.6f}")

        # ── Model 6: just terrain at different radii (no food, no pop) ────
        X6 = np.column_stack([
            np.ones(n),
            arr[:, 3] + arr[:, 4], arr[:, 5], arr[:, 6], arr[:, 7],
            arr[:, 9], arr[:, 10], arr[:, 11], arr[:, 12],
            arr[:, 13], arr[:, 14], arr[:, 15], arr[:, 16],
        ])
        c6, r2_6, mae6, _ = fit_ols(X6, food_delta)
        print(f"  Terrain only (R1+R2+R3, no food/pop): R²={r2_6:.6f}")

        print()


if __name__ == "__main__":
    main()
