"""
Food analysis v10: Narrowing down the functional form.

Key findings so far:
- terrain*(1-food) is very significant → production scales with (1-food)
- fo/pl ratio ≈ 1.5 is stable
- Residual patterns with pop suggest nonlinear pop effect
- food + food² beyond terrain*(1-food) still helps → additional food feedback?

Reproduce:  python3 data-analysis/analyze_food_v10.py
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


def collect_rows(replays):
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

                adj = terrain_adj(grid, x, y)
                food_delta = sa["food"] - sb["food"]
                pop_delta = sa["population"] - sb["population"]
                rows.append((
                    sb["population"], sb["food"],
                    *adj,
                    food_delta,
                    pop_delta,
                    sa["population"],
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

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_rows(replays)
        n = len(arr)

        pop = arr[:, 0]
        food = arr[:, 1]
        n_pl, n_fo, n_mt, n_oc = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        food_delta = arr[:, 6]
        pop_delta = arr[:, 7]
        pop_after = arr[:, 8]

        omf = 1 - food  # one minus food

        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}  ({n} obs)")
        print(f"{'=' * 72}")

        # ── Test: is pop_delta related to pop threshold? ─────────────────
        # Check if growth occurs (pop_delta < 0 means settlement split)
        grew = pop_delta < -0.05
        print(f"  Growth events: {grew.sum()} / {n} ({100*grew.mean():.1f}%)")
        if grew.sum() > 0:
            print(f"    pop_before when grew: mean={pop[grew].mean():.3f}, "
                  f"min={pop[grew].min():.3f}, max={pop[grew].max():.3f}")
            print(f"    pop_before no growth: mean={pop[~grew].mean():.3f}")

        # ── Model 1: Just terrain*(1-food) + pop ─────────────────────────
        features1 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            food_delta,
        ])
        dm1 = demean_by_group(features1, groups)
        c1, r2_1, mae1, res1 = fit_ols(dm1[:, :5], dm1[:, 5])
        print(f"\n  M1  terrain*(1-f) + pop:           R²={r2_1:.6f}")
        names1 = ["pl*(1-f)", "fo*(1-f)", "mt*(1-f)", "oc*(1-f)", "pop"]
        for nm, c in zip(names1, c1):
            print(f"       {nm:>12s} = {c:>12.6f}")

        # ── Model 2: terrain*(1-food) + pop + food*(1-food) ──────────────
        fomf = food * omf  # food*(1-food) — logistic feedback
        features2 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            fomf,
            food_delta,
        ])
        dm2 = demean_by_group(features2, groups)
        c2, r2_2, mae2, res2 = fit_ols(dm2[:, :6], dm2[:, 6])
        print(f"  M2  + food*(1-food):               R²={r2_2:.6f}  [f*(1-f)={c2[5]:.6f}]")

        # ── Model 3: terrain*(1-food) + pop + food + food² ───────────────
        features3 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            food, food**2,
            food_delta,
        ])
        dm3 = demean_by_group(features3, groups)
        c3, r2_3, _, _ = fit_ols(dm3[:, :7], dm3[:, 7])
        print(f"  M3  + food + food²:                R²={r2_3:.6f}  [f={c3[5]:.6f}, f²={c3[6]:.6f}]")

        # ── Model 4: terrain*(1-food) + pop*(1-food) ─────────────────────
        # Maybe consumption also scales with (1-food)? Or pop*food?
        features4 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop * food,  # consumption proportional to pop*food
            food_delta,
        ])
        dm4 = demean_by_group(features4, groups)
        c4, r2_4, _, _ = fit_ols(dm4[:, :5], dm4[:, 5])
        print(f"  M4  terrain*(1-f) + pop*food:      R²={r2_4:.6f}")

        # ── Model 5: (terrain + base)*(1-food) - pop_rate*pop ────────────
        # This is: base*(1-food) + terrain*(1-food) - pop*rate
        # base*(1-food) after demeaning = -base*food_dm = base*(omf)_dm
        features5 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            omf,  # base * (1-food) — captures a constant production rate scaled by (1-food)
            pop,
            food_delta,
        ])
        dm5 = demean_by_group(features5, groups)
        c5, r2_5, mae5, res5 = fit_ols(dm5[:, :6], dm5[:, 6])
        print(f"  M5  (terrain+base)*(1-f) + pop:    R²={r2_5:.6f}  [base*(1-f)={c5[4]:.6f}]")

        # ── Model 6: terrain*(1-food) + pop + pop² ───────────────────────
        features6 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop, pop**2,
            food_delta,
        ])
        dm6 = demean_by_group(features6, groups)
        c6, r2_6, _, _ = fit_ols(dm6[:, :6], dm6[:, 6])
        print(f"  M6  terrain*(1-f) + pop + pop²:    R²={r2_6:.6f}  [pop²={c6[5]:.6f}]")

        # ── Model 7: terrain*(1-food) + pop + indicator(grew) ────────────
        features7 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop,
            grew.astype(float),
            food_delta,
        ])
        dm7 = demean_by_group(features7, groups)
        c7, r2_7, _, _ = fit_ols(dm7[:, :6], dm7[:, 6])
        print(f"  M7  + grew indicator:              R²={r2_7:.6f}  [grew={c7[5]:.6f}]")

        # ── Model 8: terrain*(1-food) + pop + food*pop ───────────────────
        features8 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            pop, food * pop,
            food_delta,
        ])
        dm8 = demean_by_group(features8, groups)
        c8, r2_8, _, _ = fit_ols(dm8[:, :6], dm8[:, 6])
        print(f"  M8  terrain*(1-f) + pop + f*pop:   R²={r2_8:.6f}  [f*pop={c8[5]:.6f}]")

        # ── Model 9: Full kitchen sink ───────────────────────────────────
        features9 = np.column_stack([
            n_pl * omf, n_fo * omf, n_mt * omf, n_oc * omf,
            omf,          # base*(1-f)
            pop,
            food * pop,   # consumption interaction
            grew.astype(float),
            food_delta,
        ])
        dm9 = demean_by_group(features9, groups)
        c9, r2_9, mae9, res9 = fit_ols(dm9[:, :8], dm9[:, 8])
        print(f"  M9  full:                          R²={r2_9:.6f}, MAE={mae9:.6f}")
        names9 = ["pl*(1-f)", "fo*(1-f)", "mt*(1-f)", "oc*(1-f)", "base*(1-f)", "pop", "f*pop", "grew"]
        for nm, c in zip(names9, c9):
            print(f"       {nm:>12s} = {c:>12.6f}")

        # ── Model 10: All terrain types combined into one score ──────────
        # Check if fo = 1.5*pl always
        ratio = c1[1] / c1[0] if abs(c1[0]) > 1e-8 else float('nan')
        print(f"\n  fo/pl ratio: {ratio:.4f}")

        # Try: terrain_score = n_pl + ratio*n_fo, then terrain_score*(1-food) + pop
        terrain_score = n_pl + ratio * n_fo
        features10 = np.column_stack([
            terrain_score * omf,
            pop,
            food_delta,
        ])
        dm10 = demean_by_group(features10, groups)
        c10, r2_10, _, _ = fit_ols(dm10[:, :2], dm10[:, 2])
        print(f"  M10 score*(1-f) + pop:             R²={r2_10:.6f}  (score=pl+{ratio:.2f}*fo)")

        # ── Residual analysis for best model (M5 or M9) ──────────────────
        best_res = res5
        print(f"\n  Residual analysis of M5:")
        food_bins = np.digitize(food, np.arange(0, 1.05, 0.1))
        for b in range(1, 11):
            mask = food_bins == b
            if mask.sum() > 20:
                print(f"    food [{(b-1)*0.1:.1f}-{b*0.1:.1f}]: n={mask.sum():5d}, "
                      f"mean_res={best_res[mask].mean():>9.6f}, std={best_res[mask].std():.6f}")

        pop_bins = np.digitize(pop, np.arange(0, 3.0, 0.3))
        for b in range(1, 11):
            mask = pop_bins == b
            if mask.sum() > 20:
                print(f"    pop [{(b-1)*0.3:.1f}-{b*0.3:.1f}]: n={mask.sum():5d}, "
                      f"mean_res={best_res[mask].mean():>9.6f}, std={best_res[mask].std():.6f}")

        print()


if __name__ == "__main__":
    main()
