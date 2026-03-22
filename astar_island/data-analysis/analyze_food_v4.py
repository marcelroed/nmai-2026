"""
Food analysis v4: check determinism and track individual settlement trajectories.

Reproduce:  python3 data-analysis/analyze_food_v4.py
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
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def main():
    rounds = load_replays_by_round()
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays from "
          f"{len(rounds)} rounds\n")

    # Pick one round for detailed analysis
    for rid in sorted(rounds):
        replays = rounds[rid]
        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}")
        print(f"{'=' * 72}")

        # ── 1. Check determinism: group by (pop_quantized, food_quantized,
        #       n_pl, n_fo, n_mt, n_st) and check variance of food_delta ──

        rows_by_key = defaultdict(list)

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
                    adj = terrain_adj(grid, x, y)
                    food_delta = sa["food"] - sb["food"]

                    # Use exact food and pop values as keys (3 decimal precision)
                    key = (
                        round(sb["population"], 3),
                        round(sb["food"], 3),
                        *adj,
                    )
                    rows_by_key[key].append(food_delta)

        # Find keys with multiple observations
        multi_keys = {k: v for k, v in rows_by_key.items() if len(v) >= 3}
        if multi_keys:
            spreads = []
            for k, deltas in multi_keys.items():
                arr = np.array(deltas)
                spreads.append((arr.std(), arr.max() - arr.min(), k, len(deltas)))
            spreads.sort(key=lambda x: -x[1])

            print(f"\n  Determinism check: {len(multi_keys)} unique (pop,food,terrain) combos with ≥3 obs")
            all_spreads = [s[1] for s in spreads]
            print(f"    max-min range: mean={np.mean(all_spreads):.6f}, "
                  f"median={np.median(all_spreads):.6f}, "
                  f"max={np.max(all_spreads):.6f}")

            print(f"    Top 10 highest-spread combos:")
            for std, spread, k, n in spreads[:10]:
                deltas = rows_by_key[k]
                print(f"      key={k} n={n}: deltas={[round(d,4) for d in sorted(deltas)[:8]]}"
                      f" spread={spread:.6f}")

            # How many have essentially zero spread?
            zero_spread = sum(1 for s in all_spreads if s < 0.001)
            print(f"    Combos with spread < 0.001: {zero_spread}/{len(all_spreads)} "
                  f"({100*zero_spread/len(all_spreads):.1f}%)")
        else:
            print(f"\n  No combos with ≥3 identical (pop,food,terrain)")

        # ── 2. Track individual settlements across time ────────────────────
        print(f"\n  Individual settlement trajectories (first 3 seeds):")

        for replay in replays[:1]:
            frames = replay["frames"]
            # Track all settlements across frames
            trajectories = defaultdict(list)

            for i, frame in enumerate(frames):
                grid = frame["grid"] if i == 0 else frames[i-1]["grid"]
                for s in frame["settlements"]:
                    if s["alive"]:
                        pos = (s["x"], s["y"])
                        trajectories[pos].append((
                            frame["step"], s["population"], s["food"],
                            s["defense"], s["owner_id"],
                        ))

            # Find clean trajectories (same owner, never raided, long)
            clean = []
            for pos, traj in trajectories.items():
                if len(traj) < 10:
                    continue
                owner = traj[0][4]
                if not all(t[4] == owner for t in traj):
                    continue
                # Check no defense decreases (no raids)
                never_raided = all(
                    traj[j][3] >= traj[j-1][3] - 0.001
                    for j in range(1, len(traj))
                )
                if not never_raided:
                    continue
                clean.append((pos, traj))

            print(f"    seed={replay['seed_index']}: {len(clean)} never-raided settlements "
                  f"with ≥10 frames")

            # Show a few trajectories
            for pos, traj in clean[:3]:
                print(f"\n    Settlement at {pos}:")
                grid0 = frames[0]["grid"]
                adj = terrain_adj(grid0, pos[0], pos[1])
                print(f"      terrain adj: pl={adj[0]}, fo={adj[1]}, mt={adj[2]}, st={adj[3]}")
                print(f"      {'step':>4s} {'pop':>7s} {'food':>7s} {'def':>7s} {'Δfood':>8s}")
                for j in range(len(traj)):
                    step, pop, food, defense, _ = traj[j]
                    if j > 0:
                        delta_food = food - traj[j-1][2]
                        print(f"      {step:4d} {pop:7.3f} {food:7.3f} {defense:7.3f} {delta_food:8.4f}")
                    else:
                        print(f"      {step:4d} {pop:7.3f} {food:7.3f} {defense:7.3f}")

        # ── 3. Within-round: compare seeds at same position ────────────────
        # Settlements at the same position in different seeds have same terrain
        # but potentially different pop/food trajectories.
        # If food formula is deterministic, same (pop, food, terrain) → same delta.

        print()


if __name__ == "__main__":
    main()
