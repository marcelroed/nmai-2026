"""
Extract exact coefficients by pairing settlements with matched variables.

Within a (seed, step) group, weather is constant. If two settlements have
the same terrain and food but different pop, the food_delta difference
is PURELY due to pop.

Strategy:
1. Group observations by (seed, step, terrain_combo, food_int)
2. Within each group, differences isolate the pop coefficient
3. Similarly, group by (seed, step, food_int, pop_int) to isolate terrain

Reproduce:  python3 data-analysis/pair_analysis.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


def terrain_adj_full(grid, x, y):
    h, w = len(grid), len(grid[0])
    counts = defaultdict(int)
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if nx == x and ny == y:
                continue
            counts[grid[ny][nx]] += 1
    return dict(counts)


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


def collect_all_clean(replays):
    """Collect all clean observations with full detail."""
    rows = []
    for replay in replays:
        frames = replay["frames"]
        seed = replay["seed_index"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            grid_before = fb["grid"]
            grid_after = fa["grid"]
            step = fa["step"]
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
                if grid_before[y][x] == PORT:
                    continue
                if sa["defense"] - sb["defense"] < -0.001:
                    continue
                if sa["food"] >= 0.997 or sa["food"] <= 0.001:
                    continue
                if sa["population"] - sb["population"] < -0.05:
                    continue
                adj_before = terrain_adj_full(grid_before, x, y)
                adj_after = terrain_adj_full(grid_after, x, y)
                if adj_before != adj_after:
                    continue
                n_pl = adj_before.get(PLAINS, 0)
                n_fo = adj_before.get(FOREST, 0)
                n_mt = adj_before.get(MOUNTAIN, 0)
                n_oc = adj_before.get(OCEAN, 0)
                n_st = adj_before.get(SETTLEMENT, 0) + adj_before.get(PORT, 0) + adj_before.get(RUIN, 0)
                food_delta = sa["food"] - sb["food"]
                food_int = round(sb["food"] * 1000)
                rows.append({
                    "seed": seed, "step": step,
                    "pop": sb["population"], "food": sb["food"], "food_int": food_int,
                    "n_pl": n_pl, "n_fo": n_fo, "n_mt": n_mt, "n_oc": n_oc, "n_st": n_st,
                    "food_delta": food_delta,
                    "terrain_key": (n_pl, n_fo, n_mt, n_oc, n_st),
                })
    return rows


def main():
    rounds = load_replays_by_round()

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Pop coefficient from matched pairs
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PART 1: Extract pop coefficient from matched pairs")
    print("  Matching: same (seed, step, terrain, food_int)")
    print("  food_delta difference = pop_rate × pop_difference")
    print("=" * 80)

    for rid in sorted(rounds):
        replays = rounds[rid]
        rows = collect_all_clean(replays)

        # Group by (seed, step, terrain, food_int)
        groups = defaultdict(list)
        for r in rows:
            key = (r["seed"], r["step"], r["terrain_key"], r["food_int"])
            groups[key].append(r)

        # Find pairs with different pop
        pop_slopes = []
        for key, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    a, b = members[i], members[j]
                    dpop = a["pop"] - b["pop"]
                    dfd = a["food_delta"] - b["food_delta"]
                    if abs(dpop) > 0.001:
                        slope = dfd / dpop
                        pop_slopes.append(slope)

        if pop_slopes:
            pop_slopes = np.array(pop_slopes)
            print(f"\n  {rid[:12]}: {len(pop_slopes)} pairs")
            print(f"    pop_rate = median {np.median(pop_slopes):.6f}, "
                  f"mean {np.mean(pop_slopes):.6f} ± {np.std(pop_slopes):.6f}")
            # Percentiles
            for p in [10, 25, 50, 75, 90]:
                print(f"    P{p:02d} = {np.percentile(pop_slopes, p):.6f}")
        else:
            print(f"\n  {rid[:12]}: no matched pairs found")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Terrain coefficient from matched pairs
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 2: Extract terrain coefficients from matched pairs")
    print("  Matching: same (seed, step, food_int)")
    print("  Difference reveals terrain × (1-f) coefficient")
    print(f"{'='*80}")

    for rid in sorted(rounds)[:3]:  # first 3 rounds for brevity
        replays = rounds[rid]
        rows = collect_all_clean(replays)

        # Group by (seed, step, food_int)
        groups = defaultdict(list)
        for r in rows:
            key = (r["seed"], r["step"], r["food_int"])
            groups[key].append(r)

        # For pairs with same pop but different terrain
        pl_diffs = []  # (Δn_pl, Δfood_delta, (1-f), Δpop)
        fo_diffs = []

        for key, members in groups.items():
            if len(members) < 2:
                continue
            food_val = key[2] / 1000.0
            omf = 1 - food_val
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    a, b = members[i], members[j]
                    dpop = a["pop"] - b["pop"]
                    dfd = a["food_delta"] - b["food_delta"]
                    dpl = a["n_pl"] - b["n_pl"]
                    dfo = a["n_fo"] - b["n_fo"]
                    dmt = a["n_mt"] - b["n_mt"]
                    doc = a["n_oc"] - b["n_oc"]
                    dst = a["n_st"] - b["n_st"]

                    # Only consider pairs where only pl changes
                    if dpl != 0 and dfo == 0 and dmt == 0 and doc == 0 and dst == 0:
                        pl_diffs.append((dpl, dfd, omf, dpop))
                    # Only fo changes
                    if dfo != 0 and dpl == 0 and dmt == 0 and doc == 0 and dst == 0:
                        fo_diffs.append((dfo, dfd, omf, dpop))

        if pl_diffs:
            arr_pl = np.array(pl_diffs)
            # food_delta_diff = pl_rate × Δn_pl × (1-f) - pop_rate × Δpop
            # If we correct for pop: (dfd + pop_rate * dpop) / (dpl * omf) = pl_rate
            # Use estimated pop_rate ≈ -0.113
            pop_rate_est = -0.113
            pl_rates = (arr_pl[:, 1] - pop_rate_est * arr_pl[:, 3]) / (arr_pl[:, 0] * arr_pl[:, 2])
            print(f"\n  {rid[:12]}: {len(pl_diffs)} pairs with only Δpl")
            print(f"    pl_rate = median {np.median(pl_rates):.6f}, "
                  f"mean {np.mean(pl_rates):.6f} ± {np.std(pl_rates):.6f}")
        else:
            print(f"\n  {rid[:12]}: no pure Δpl pairs")

        if fo_diffs:
            arr_fo = np.array(fo_diffs)
            fo_rates = (arr_fo[:, 1] - pop_rate_est * arr_fo[:, 3]) / (arr_fo[:, 0] * arr_fo[:, 2])
            print(f"    {len(fo_diffs)} pairs with only Δfo")
            print(f"    fo_rate = median {np.median(fo_rates):.6f}, "
                  f"mean {np.mean(fo_rates):.6f} ± {np.std(fo_rates):.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Within exact food match, what determines food_delta?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 3: Within exact-food groups, fit food_delta vs terrain and pop")
    print(f"{'='*80}")

    for rid in sorted(rounds):
        replays = rounds[rid]
        rows = collect_all_clean(replays)

        # Group by (seed, step, food_int)
        groups = defaultdict(list)
        for r in rows:
            key = (r["seed"], r["step"], r["food_int"])
            groups[key].append(r)

        # For groups with 5+ members, fit food_delta = a*n_pl + b*n_fo + c*pop + d
        coeffs = []
        for key, members in groups.items():
            if len(members) < 5:
                continue
            food_val = key[2] / 1000.0
            omf = 1 - food_val
            fd = np.array([m["food_delta"] for m in members])
            pl = np.array([m["n_pl"] for m in members], dtype=float)
            fo = np.array([m["n_fo"] for m in members], dtype=float)
            pop = np.array([m["pop"] for m in members])

            # Fit: food_delta = a*n_pl + b*n_fo + c*pop + intercept
            X = np.column_stack([pl, fo, pop, np.ones(len(fd))])
            try:
                c, _, _, _ = np.linalg.lstsq(X, fd, rcond=None)
                if not np.any(np.isnan(c)):
                    # c[0] should be pl_rate * (1-f), c[1] = fo_rate * (1-f), c[2] = -pop_rate
                    pl_r = c[0] / omf if omf > 0.01 else np.nan
                    fo_r = c[1] / omf if omf > 0.01 else np.nan
                    coeffs.append((pl_r, fo_r, c[2], food_val, len(members)))
            except:
                pass

        if coeffs:
            arr_c = np.array(coeffs)
            # Weight by group size
            weights = arr_c[:, 4]
            mask = ~np.isnan(arr_c[:, 0]) & ~np.isnan(arr_c[:, 1])
            mask &= np.abs(arr_c[:, 0]) < 0.2  # remove crazy values
            mask &= np.abs(arr_c[:, 1]) < 0.2
            if mask.sum() > 5:
                w = weights[mask]
                pl_avg = np.average(arr_c[mask, 0], weights=w)
                fo_avg = np.average(arr_c[mask, 1], weights=w)
                pop_avg = np.average(arr_c[mask, 2], weights=w)
                print(f"\n  {rid[:12]}: {mask.sum()} valid groups")
                print(f"    Weighted avg: pl_rate={pl_avg:.6f} fo_rate={fo_avg:.6f} pop_rate={-pop_avg:.6f}")
                if abs(pl_avg) > 1e-6:
                    print(f"    fo/pl = {fo_avg/pl_avg:.4f}")


if __name__ == "__main__":
    main()
