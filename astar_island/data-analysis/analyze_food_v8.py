"""
Food analysis v8: Check data precision, and test if food_delta is stochastic.

Reproduce:  python3 data-analysis/analyze_food_v8.py
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


def main():
    rounds = load_replays_by_round()

    # ── 1. Check food value precision ──────────────────────────────────────
    print("=== FOOD VALUE PRECISION CHECK ===\n")
    sample = list(rounds.values())[0][0]
    food_vals = []
    for frame in sample["frames"]:
        for s in frame["settlements"]:
            food_vals.append(s["food"])

    food_vals = np.array(food_vals)
    # Check if values are exact multiples of 0.001
    remainders = (food_vals * 1000) % 1
    print(f"  Sample: {len(food_vals)} food values")
    print(f"  Multiples of 0.001: {np.sum(np.abs(remainders) < 1e-8)} / {len(food_vals)}")
    print(f"  Max remainder: {remainders.max():.10f}")
    # Check unique decimal places
    unique_foods = np.unique(np.round(food_vals, 6))
    print(f"  Unique food values: {len(unique_foods)}")
    print(f"  Sample values: {unique_foods[:20]}")
    print()

    # ── 2. For one round: same map, different seeds ────────────────────────
    # At step 0→1, same position should have same terrain.
    # Track food_delta at each position across seeds.
    print("=== SAME POSITION ACROSS SEEDS (STEP 0→1) ===\n")

    for rid in sorted(rounds):
        replays = rounds[rid]
        if len(replays) < 3:
            continue

        pos_data = defaultdict(list)  # pos -> list of (pop, food, wealth, def, food_delta, seed)

        for replay in replays:
            fb, fa = replay["frames"][0], replay["frames"][1]
            grid = fb["grid"]
            before_map = {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}
            after_map = {(s["x"], s["y"]): s for s in fa["settlements"] if s["alive"]}

            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                food_delta = sa["food"] - sb["food"]
                pos_data[pos].append((
                    sb["population"], sb["food"], sb["wealth"], sb["defense"],
                    food_delta, replay["seed_index"],
                ))

        # Positions present in ALL seeds
        n_seeds = len(replays)
        common_pos = {k: v for k, v in pos_data.items() if len(v) == n_seeds}

        if not common_pos:
            continue

        print(f"Round {rid[:12]}: {len(common_pos)} positions in all {n_seeds} seeds")

        # For each common position, show the data
        spreads = []
        for pos in sorted(common_pos)[:10]:
            data = common_pos[pos]
            x, y = pos
            adj = terrain_adj(replays[0]["frames"][0]["grid"], x, y)
            deltas = [d[4] for d in data]
            pops = [d[0] for d in data]
            foods = [d[1] for d in data]
            spread = max(deltas) - min(deltas)
            spreads.append(spread)
            print(f"  pos=({x:2d},{y:2d}) adj=({adj[0]},{adj[1]},{adj[2]},{adj[3]})")
            for pop, food, w, d, fd, seed in sorted(data, key=lambda x: x[5]):
                print(f"    seed={seed}: pop={pop:.3f} food={food:.3f} w={w:.3f} d={d:.3f} → Δfood={fd:.3f}")

        if spreads:
            print(f"  spread stats: mean={np.mean(spreads):.4f}, max={np.max(spreads):.4f}")

        # ── 3. At same position, do differences in pop/food explain delta? ──
        print(f"\n  Testing if Δfood = a + b*pop + c*food explains per-position variation:")
        all_diffs = []
        for pos, data in common_pos.items():
            if len(data) < 3:
                continue
            # Within this position, terrain is identical.
            # If food_delta = const + b*pop + c*food, then:
            # Δfood_i - Δfood_j = b*(pop_i - pop_j) + c*(food_i - food_j)
            for i in range(len(data)):
                for j in range(i+1, len(data)):
                    dpop = data[i][0] - data[j][0]
                    dfood = data[i][1] - data[j][1]
                    ddelta = data[i][4] - data[j][4]
                    all_diffs.append((dpop, dfood, ddelta))

        if all_diffs:
            diffs = np.array(all_diffs)
            # Fit ddelta = b*dpop + c*dfood (no intercept)
            X = diffs[:, :2]
            y = diffs[:, 2]
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coeffs
            res = y - pred
            ss_res = np.sum(res**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
            print(f"    Δ(Δfood) = {coeffs[0]:.4f}*Δpop + {coeffs[1]:.4f}*Δfood, "
                  f"R²={r2:.4f}")
            print(f"    residual std={res.std():.6f}, MAE={np.mean(np.abs(res)):.6f}")
            print(f"    This tells us how much of the food_delta variation at the same")
            print(f"    position is explained by pop/food initial differences")

        print()
        break  # Only show first round in detail


if __name__ == "__main__":
    main()
