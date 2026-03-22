"""
Food analysis v5: Focus on step 0→1 (cleanest possible data) and check
whether the grid at each step (not initial) explains food_delta better.

Reproduce:  python3 data-analysis/analyze_food_v5.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


def terrain_adj(grid, x, y):
    h, w = len(grid), len(grid[0])
    np_, nf, nm, ns = 0, 0, 0, 0
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if nx == x and ny == y:
                continue
            t = grid[ny][nx]
            if t in (PLAINS, EMPTY):
                np_ += 1
            elif t == FOREST:
                nf += 1
            elif t == MOUNTAIN:
                nm += 1
            elif t in (SETTLEMENT, PORT, RUIN):
                ns += 1
    return np_, nf, nm, ns


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
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays from "
          f"{len(rounds)} rounds\n")

    for rid in sorted(rounds):
        replays = rounds[rid]
        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}")
        print(f"{'=' * 72}")

        # ── Step 0→1 only ──────────────────────────────────────────────────
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
                adj = terrain_adj(grid, x, y)
                rows.append((
                    sb["population"], sb["food"], sb["wealth"],
                    sa["population"], sa["food"], sa["wealth"],
                    *adj,
                    replay["seed_index"],
                ))

        arr = np.array(rows, dtype=np.float64)
        pop_b = arr[:, 0]
        food_b = arr[:, 1]
        food_a = arr[:, 4]
        food_delta = food_a - food_b
        n_pl, n_fo, n_mt, n_st = arr[:, 6], arr[:, 7], arr[:, 8], arr[:, 9]

        n = len(arr)
        print(f"  Step 0→1: {n} settlements")
        print(f"    pop_b: mean={pop_b.mean():.3f}, std={pop_b.std():.3f}")
        print(f"    food_b: mean={food_b.mean():.3f}, std={food_b.std():.3f}")
        print(f"    food_delta: mean={food_delta.mean():.4f}, std={food_delta.std():.4f}")

        # Check determinism at step 0→1
        from collections import Counter
        key_groups = defaultdict(list)
        for i in range(n):
            key = (round(pop_b[i], 3), round(food_b[i], 3),
                   int(n_pl[i]), int(n_fo[i]), int(n_mt[i]), int(n_st[i]))
            key_groups[key].append(food_delta[i])

        multi = {k: v for k, v in key_groups.items() if len(v) >= 2}
        if multi:
            spreads = [max(v) - min(v) for v in multi.values()]
            print(f"\n    Determinism (step 0→1): {len(multi)} combos with ≥2 obs")
            print(f"      max-min spread: mean={np.mean(spreads):.6f}, "
                  f"median={np.median(spreads):.6f}, max={np.max(spreads):.6f}")
            zero_spread = sum(1 for s in spreads if s < 0.001)
            print(f"      spread < 0.001: {zero_spread}/{len(multi)} ({100*zero_spread/len(multi):.1f}%)")

            # Show worst
            worst = sorted(multi.items(), key=lambda x: -(max(x[1])-min(x[1])))[:5]
            for k, vs in worst:
                print(f"      key={k}: deltas={[round(v,4) for v in sorted(vs)]}")

        # Fit linear model on step 0→1
        X = np.column_stack([
            np.ones(n), food_b, pop_b, n_pl, n_fo, n_mt, n_st,
        ])
        c, r2, mae, res = fit_ols(X, food_delta)
        print(f"\n    Linear fit (int+food+pop+pl+fo+mt+st): R²={r2:.6f}, MAE={mae:.6f}")
        names = ["intercept", "food_b", "pop_b", "n_pl", "n_fo", "n_mt", "n_st"]
        for name, coeff in zip(names, c):
            print(f"      {name:>12s} = {coeff:>10.6f}")

        # ── Compare step 0→1 with later steps (same terrain, same position)
        print(f"\n    Per-step food_delta for similar pop/terrain bins:")
        # Collect per step
        step_deltas = defaultdict(list)
        for replay in replays:
            frames = replay["frames"]
            for i in range(len(frames) - 1):
                fb, fa = frames[i], frames[i + 1]
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
                    if sa["defense"] - sb["defense"] < -0.001:
                        continue
                    x, y = pos
                    if grid[y][x] == PORT:
                        continue
                    # Control for similar conditions: pop~1.0, food~0.5, pl+fo >= 3
                    adj = terrain_adj(grid, x, y)
                    if (0.8 <= sb["population"] <= 1.2 and
                        0.4 <= sb["food"] <= 0.6 and
                        adj[0] + adj[1] >= 3):
                        step_deltas[fa["step"]].append(sa["food"] - sb["food"])

        if step_deltas:
            print(f"      (controlled: pop∈[0.8,1.2], food∈[0.4,0.6], pl+fo≥3)")
            for step in sorted(step_deltas.keys()):
                deltas = step_deltas[step]
                if len(deltas) >= 5:
                    arr_d = np.array(deltas)
                    print(f"      step {step:2d}: n={len(deltas):4d}, "
                          f"mean={arr_d.mean():>8.4f}, std={arr_d.std():>7.4f}")

        print()


if __name__ == "__main__":
    main()
