"""
Food analysis v11: Separate growth from production.

Fit on non-growth observations only to cleanly estimate the food production
formula. Then characterize the growth food cost separately.

Reproduce:  python3 data-analysis/analyze_food_v11.py
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


def collect_rows(replays, exclude_growth=False):
    rows = []
    groups = []
    for replay in replays:
        frames = replay["frames"]
        seed = replay["seed_index"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            grid = fb["grid"]
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
                if sa["defense"] - sb["defense"] < -0.001:
                    continue  # raided
                x, y = pos
                if grid[y][x] == PORT:
                    continue
                if sa["food"] >= 0.997 or sa["food"] <= 0.001:
                    continue

                pop_delta = sa["population"] - sb["population"]
                grew = pop_delta < -0.05

                if exclude_growth and grew:
                    continue

                adj = terrain_adj(grid, x, y)
                food_delta = sa["food"] - sb["food"]
                rows.append((
                    sb["population"], sb["food"],
                    *adj,
                    food_delta,
                    float(grew),
                    pop_delta,
                ))
                groups.append((seed, step))

    return np.array(rows, dtype=np.float64), groups


def fit_ols(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    res = y - pred
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def demean_by_group(arr, groups):
    group_ids = {}
    for i, g in enumerate(groups):
        if g not in group_ids:
            group_ids[g] = []
        group_ids[g].append(i)
    result = arr.copy()
    for indices in group_ids.values():
        group_mean = arr[indices].mean(axis=0)
        result[indices] -= group_mean
    return result


def main():
    rounds = load_replays_by_round()

    print("=" * 72)
    print("PART 1: FIT ON NON-GROWTH OBSERVATIONS ONLY")
    print("=" * 72)

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_rows(replays, exclude_growth=True)
        n = len(arr)

        pop = arr[:, 0]
        food = arr[:, 1]
        n_pl, n_fo, n_mt, n_oc = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        food_delta = arr[:, 6]
        omf = 1 - food

        print(f"\nRound {rid[:12]}  ({n} non-growth obs)")

        # ── M1: terrain*(1-food) + pop ───────────────────────────────────
        features = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            food_delta,
        ])
        dm = demean_by_group(features, groups)
        c1, r2_1, mae1, res1 = fit_ols(dm[:, :5], dm[:, 5])
        print(f"  M1 terrain*(1-f) + pop:         R²={r2_1:.6f}, MAE={mae1:.6f}")
        names = ["pl*(1-f)", "fo*(1-f)", "mt*(1-f)", "oc*(1-f)", "pop"]
        for nm, c in zip(names, c1):
            print(f"     {nm:>12s} = {c:>12.8f}")

        # fo/pl ratio
        ratio = c1[1] / c1[0] if abs(c1[0]) > 1e-8 else float('nan')
        print(f"     fo/pl ratio: {ratio:.4f}")

        # ── M2: + food*(1-food) ──────────────────────────────────────────
        features2 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            food * omf,
            food_delta,
        ])
        dm2 = demean_by_group(features2, groups)
        c2, r2_2, mae2, res2 = fit_ols(dm2[:, :6], dm2[:, 6])
        print(f"  M2 + food*(1-f):                R²={r2_2:.6f}  [f*(1-f)={c2[5]:.6f}]")

        # ── M3: + food + food² ───────────────────────────────────────────
        features3 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            food, food**2,
            food_delta,
        ])
        dm3 = demean_by_group(features3, groups)
        c3, r2_3, mae3, _ = fit_ols(dm3[:, :7], dm3[:, 7])
        print(f"  M3 + food + food²:              R²={r2_3:.6f}  [f={c3[5]:.6f}, f²={c3[6]:.6f}]")

        # ── Residual analysis for M1 ─────────────────────────────────────
        food_bins = np.digitize(food, np.arange(0, 1.05, 0.1))
        print(f"  M1 residuals by food:")
        for b in range(1, 11):
            mask = food_bins == b
            if mask.sum() > 20:
                print(f"    [{(b-1)*0.1:.1f}-{b*0.1:.1f}] n={mask.sum():5d} mean_res={res1[mask].mean():>9.6f}")

        pop_bins = np.digitize(pop, [0.5, 0.8, 1.0, 1.2, 1.5])
        print(f"  M1 residuals by pop:")
        for b in range(6):
            mask = pop_bins == b
            if mask.sum() > 20:
                labels = ["<0.5", "0.5-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", ">1.5"]
                print(f"    [{labels[b]:>7s}] n={mask.sum():5d} mean_res={res1[mask].mean():>9.6f}")

    print("\n\n" + "=" * 72)
    print("PART 2: GROWTH FOOD COST ANALYSIS")
    print("=" * 72)

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr_all, groups_all = collect_rows(replays, exclude_growth=False)
        arr_ng, groups_ng = collect_rows(replays, exclude_growth=True)

        grew_mask = arr_all[:, 7] > 0.5
        n_grew = grew_mask.sum()
        n_total = len(arr_all)

        if n_grew < 10:
            print(f"\nRound {rid[:12]}: {n_grew} growth events (too few)")
            continue

        print(f"\nRound {rid[:12]}: {n_grew} growth events / {n_total} total")

        # Growth events only
        g_pop = arr_all[grew_mask, 0]
        g_food = arr_all[grew_mask, 1]
        g_fd = arr_all[grew_mask, 6]
        g_pd = arr_all[grew_mask, 8]  # pop_delta

        # Use the non-growth model to predict what food_delta SHOULD have been
        # without growth, then the difference is the growth cost
        pop = arr_ng[:, 0]
        food = arr_ng[:, 1]
        omf = 1 - food
        n_pl, n_fo, n_mt, n_oc = arr_ng[:, 2], arr_ng[:, 3], arr_ng[:, 4], arr_ng[:, 5]

        features = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            arr_ng[:, 6],
        ])
        dm = demean_by_group(features, groups_ng)
        c, _, _, _ = fit_ols(dm[:, :5], dm[:, 5])

        # Now predict for growth events using group means from non-growth
        # Actually, better: fit model on ALL data with grew indicator
        # The grew coefficient IS the growth cost
        pop_all = arr_all[:, 0]
        food_all = arr_all[:, 1]
        omf_all = 1 - food_all
        n_pl_a = arr_all[:, 2]
        n_fo_a = arr_all[:, 3]
        n_mt_a = arr_all[:, 4]
        n_oc_a = arr_all[:, 5]

        features_all = np.column_stack([
            n_pl_a * omf_all, n_fo_a * omf_all,
            n_mt_a * omf_all, n_oc_a * omf_all,
            pop_all,
            grew_mask.astype(float),
            arr_all[:, 6],
        ])
        dm_all = demean_by_group(features_all, groups_all)
        c_all, r2_all, _, _ = fit_ols(dm_all[:, :6], dm_all[:, 6])
        print(f"  grew coefficient: {c_all[5]:.6f}")

        # Is growth cost proportional to food?
        features_gf = np.column_stack([
            n_pl_a * omf_all, n_fo_a * omf_all,
            n_mt_a * omf_all, n_oc_a * omf_all,
            pop_all,
            grew_mask.astype(float) * food_all,  # growth_cost * food
            arr_all[:, 6],
        ])
        dm_gf = demean_by_group(features_gf, groups_all)
        c_gf, r2_gf, _, _ = fit_ols(dm_gf[:, :6], dm_gf[:, 6])
        print(f"  grew*food coefficient: {c_gf[5]:.6f}  (R²={r2_gf:.6f} vs {r2_all:.6f})")

        # Try both grew and grew*food
        features_both = np.column_stack([
            n_pl_a * omf_all, n_fo_a * omf_all,
            n_mt_a * omf_all, n_oc_a * omf_all,
            pop_all,
            grew_mask.astype(float),
            grew_mask.astype(float) * food_all,
            arr_all[:, 6],
        ])
        dm_both = demean_by_group(features_both, groups_all)
        c_both, r2_both, _, _ = fit_ols(dm_both[:, :7], dm_both[:, 7])
        print(f"  grew + grew*food: grew={c_both[5]:.6f}, grew*food={c_both[6]:.6f}  R²={r2_both:.6f}")

        # Characterize growth events
        print(f"  Growth event stats:")
        print(f"    pop_before: mean={g_pop.mean():.3f}, min={g_pop.min():.3f}, max={g_pop.max():.3f}")
        print(f"    food_before: mean={g_food.mean():.3f}")
        print(f"    pop_delta: mean={g_pd.mean():.3f} (child takes pop)")

    print("\n\n" + "=" * 72)
    print("PART 3: COEFFICIENT STABILITY (per-seed, non-growth)")
    print("=" * 72)

    for rid in sorted(rounds):
        replays = rounds[rid]
        print(f"\nRound {rid[:12]}:")

        # Per-seed coefficients
        for replay in replays:
            seed_arr, seed_groups = collect_rows([replay], exclude_growth=True)
            if len(seed_arr) < 50:
                continue
            pop = seed_arr[:, 0]
            food = seed_arr[:, 1]
            omf = 1 - food
            features = np.column_stack([
                seed_arr[:, 2] * omf, seed_arr[:, 3] * omf,
                seed_arr[:, 4] * omf, seed_arr[:, 5] * omf,
                pop,
                seed_arr[:, 6],
            ])
            dm = demean_by_group(features, seed_groups)
            c, r2, _, _ = fit_ols(dm[:, :5], dm[:, 5])
            fo_pl = c[1] / c[0] if abs(c[0]) > 1e-6 else float('nan')
            print(f"  seed={replay['seed_index']}: R²={r2:.4f}  "
                  f"pl={c[0]:.6f} fo={c[1]:.6f} mt={c[2]:.6f} oc={c[3]:.6f} "
                  f"pop={c[4]:.6f} fo/pl={fo_pl:.3f}")


if __name__ == "__main__":
    main()
