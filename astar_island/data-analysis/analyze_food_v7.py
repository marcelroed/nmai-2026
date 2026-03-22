"""
Food analysis v7: Per-seed fit at step 0→1 to check for randomness,
and try wealth/defense as predictors.

Reproduce:  python3 data-analysis/analyze_food_v7.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


def terrain_adj(grid, x, y):
    h, w = len(grid), len(grid[0])
    np_, nf, nm, oc = 0, 0, 0, 0
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if nx == x and ny == y:
                continue
            t = grid[ny][nx]
            if t == PLAINS:
                np_ += 1
            elif t == FOREST:
                nf += 1
            elif t == MOUNTAIN:
                nm += 1
            elif t == OCEAN:
                oc += 1
            # Settlement/Port/Ruin: not counted (will be 0 at step 0 start)
    return np_, nf, nm, oc


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
    max_err = np.max(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def main():
    rounds = load_replays_by_round()
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays\n")

    for rid in sorted(rounds):
        replays = rounds[rid]
        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}")
        print(f"{'=' * 72}")

        # ── Per-seed step 0→1 ──────────────────────────────────────────────
        for replay in replays[:2]:  # First 2 seeds
            frames = replay["frames"]
            fb, fa = frames[0], frames[1]
            grid = fb["grid"]
            before_map = {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}
            after_map = {(s["x"], s["y"]): s for s in fa["settlements"] if s["alive"]}

            rows = []
            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                adj = terrain_adj(grid, x, y)
                rows.append((
                    sb["population"], sb["food"], sb["wealth"], sb["defense"],
                    sa["food"],
                    *adj,
                    x, y,
                ))

            arr = np.array(rows, dtype=np.float64)
            food_delta = arr[:, 4] - arr[:, 1]
            n = len(arr)

            # Model A: food + pop + terrain (pl, fo)
            X_a = np.column_stack([
                np.ones(n), arr[:, 1], arr[:, 0],
                arr[:, 5], arr[:, 6],  # n_plains, n_forest
            ])
            _, r2_a, mae_a, _ = fit_ols(X_a, food_delta)

            # Model B: + mountain + ocean
            X_b = np.column_stack([
                np.ones(n), arr[:, 1], arr[:, 0],
                arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8],
            ])
            c_b, r2_b, mae_b, _ = fit_ols(X_b, food_delta)

            # Model C: + wealth + defense
            X_c = np.column_stack([
                np.ones(n), arr[:, 1], arr[:, 0], arr[:, 2], arr[:, 3],
                arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8],
            ])
            c_c, r2_c, mae_c, res_c = fit_ols(X_c, food_delta)

            print(f"\n  seed={replay['seed_index']} ({n} settlements)")
            print(f"    A (food+pop+pl+fo):            R²={r2_a:.6f}, MAE={mae_a:.6f}")
            print(f"    B (+mt+oc):                    R²={r2_b:.6f}, MAE={mae_b:.6f}")
            print(f"    C (+wealth+defense):            R²={r2_c:.6f}, MAE={mae_c:.6f}")
            names = ["int", "food", "pop", "wealth", "def", "pl", "fo", "mt", "oc"]
            coeffs_str = ", ".join(f"{n}={c:.4f}" for n, c in zip(names, c_c))
            print(f"      [{coeffs_str}]")

            # Show worst residuals
            worst = np.argsort(np.abs(res_c))[-3:]
            print(f"    Worst residuals from model C:")
            for idx in worst:
                r = arr[idx]
                print(f"      pos=({int(r[9])},{int(r[10])}) pop={r[0]:.3f} food={r[1]:.3f} "
                      f"w={r[2]:.3f} d={r[3]:.3f} "
                      f"terrain=({int(r[5])},{int(r[6])},{int(r[7])},{int(r[8])}) "
                      f"delta={food_delta[idx]:.4f} pred={food_delta[idx]-res_c[idx]:.4f} "
                      f"res={res_c[idx]:.4f}")

        # ── All seeds combined, step 0→1, with x,y features ───────────────
        all_rows = []
        for replay in replays:
            fb, fa = replay["frames"][0], replay["frames"][1]
            grid = fb["grid"]
            before_map = {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}
            after_map = {(s["x"], s["y"]): s for s in fa["settlements"] if s["alive"]}
            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                adj = terrain_adj(grid, x, y)
                all_rows.append((
                    sb["population"], sb["food"], sb["wealth"], sb["defense"],
                    sa["food"],
                    *adj, x, y,
                ))

        arr = np.array(all_rows, dtype=np.float64)
        food_delta = arr[:, 4] - arr[:, 1]
        n = len(arr)

        # All seeds: with position features
        X = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0], arr[:, 2], arr[:, 3],
            arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8],
        ])
        _, r2, mae, _ = fit_ols(X, food_delta)

        # Try adding x, y
        X_xy = np.column_stack([
            np.ones(n), arr[:, 1], arr[:, 0], arr[:, 2], arr[:, 3],
            arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8],
            arr[:, 9], arr[:, 10],
        ])
        _, r2_xy, mae_xy, _ = fit_ols(X_xy, food_delta)

        print(f"\n  All seeds step 0→1 ({n} obs):")
        print(f"    with food+pop+w+d+terrain: R²={r2:.6f}")
        print(f"    + x,y position:            R²={r2_xy:.6f}")

        # ── Check if same position across seeds gives same food_delta ──────
        pos_deltas = defaultdict(list)
        for row in all_rows:
            pos = (int(row[9]), int(row[10]))
            pos_deltas[pos].append(row[4] - row[1])  # food_delta

        multi_pos = {k: v for k, v in pos_deltas.items() if len(v) >= 3}
        if multi_pos:
            spreads = [max(v) - min(v) for v in multi_pos.values()]
            print(f"\n    Same position across seeds ({len(multi_pos)} positions with ≥3 obs):")
            print(f"      spread: mean={np.mean(spreads):.6f}, max={np.max(spreads):.6f}")
            zero = sum(1 for s in spreads if s < 0.005)
            print(f"      spread < 0.005: {zero}/{len(multi_pos)} ({100*zero/len(multi_pos):.1f}%)")

            # Show some examples
            worst_pos = sorted(multi_pos.items(), key=lambda x: -(max(x[1])-min(x[1])))[:5]
            for pos, ds in worst_pos:
                print(f"      pos={pos}: deltas={[round(d,4) for d in sorted(ds)]}")

        print()


if __name__ == "__main__":
    main()
