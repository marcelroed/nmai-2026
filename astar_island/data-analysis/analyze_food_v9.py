"""
Food analysis v9: Explore nonlinear functional forms after discovering
that food² is significant. Test candidate formulas.

Reproduce:  python3 data-analysis/analyze_food_v9.py
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
                rows.append((
                    sb["population"], sb["food"],
                    *adj,
                    food_delta,
                ))
                groups.append((seed, step))

    return np.array(rows, dtype=np.float64), groups


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
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays from "
          f"{len(rounds)} rounds\n")

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_rows(replays)
        n = len(arr)

        pop = arr[:, 0]
        food = arr[:, 1]
        n_pl, n_fo, n_mt, n_oc = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        food_delta = arr[:, 6]

        # Compute candidate nonlinear features
        food_sq = food ** 2
        one_minus_food = 1 - food
        food_times_omf = food * one_minus_food  # food*(1-food)
        pop_sq = pop ** 2
        pop_food = pop * food

        # Terrain "score" = weighted sum
        # Let's also try terrain * (1-food) interactions
        terrain_basic = n_pl + n_fo  # simplified terrain
        terrain_omf = terrain_basic * one_minus_food

        n_groups = len(set(groups))

        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}  ({n} obs, {n_groups} groups)")
        print(f"{'=' * 72}")

        # All candidate features, then demean
        all_features = np.column_stack([
            food, pop, n_pl, n_fo, n_mt, n_oc,           # 0-5: basic
            food_sq,                                       # 6
            one_minus_food,                                # 7
            food_times_omf,                                # 8: food*(1-food)
            pop_food,                                      # 9: pop*food
            pop_sq,                                        # 10
            n_pl * one_minus_food,                         # 11: plains*(1-food)
            n_fo * one_minus_food,                         # 12: forest*(1-food)
            n_mt * one_minus_food,                         # 13: mountain*(1-food)
            n_oc * one_minus_food,                         # 14: ocean*(1-food)
            food_delta,                                    # 15
        ])

        dm = demean_by_group(all_features, groups)
        y = dm[:, 15]

        # ── Model A: linear (baseline) ──────────────────────────────────────
        X_a = dm[:, :6]
        c_a, r2_a, mae_a, _ = fit_ols(X_a, y)
        print(f"  A  linear (food,pop,terrain):     R²={r2_a:.6f}")

        # ── Model B: + food² ────────────────────────────────────────────────
        X_b = dm[:, [0,1,2,3,4,5,6]]
        c_b, r2_b, mae_b, _ = fit_ols(X_b, y)
        print(f"  B  + food²:                       R²={r2_b:.6f}")
        names_b = ["food", "pop", "pl", "fo", "mt", "oc", "food²"]
        for nm, c in zip(names_b, c_b):
            print(f"       {nm:>12s} = {c:>12.6f}")

        # ── Model C: food*(1-food) instead of food + food² ─────────────────
        # food*(1-food) = food - food², so if model is a*food*(1-f), that's
        # equivalent to a*food - a*food². But if both food and food*(1-food)
        # are used, they're linearly dependent with food². Let's just try
        # replacing food with (1-food) and food*(1-food).
        X_c = dm[:, [7,1,2,3,4,5]]  # (1-food), pop, terrain
        c_c, r2_c, _, _ = fit_ols(X_c, y)
        print(f"  C  (1-food),pop,terrain:          R²={r2_c:.6f}")
        # Note: (1-food) = -food + const, so after demeaning, (1-food)_dm = -food_dm
        # So this is identical to model A. Let me verify...

        # ── Model D: food + food² + pop + terrain ───────────────────────────
        # Already done as B. Now try terrain*(1-food):
        X_d = dm[:, [0,1,6, 11,12,13,14]]  # food,pop,food², pl*(1-f), fo*(1-f), mt*(1-f), oc*(1-f)
        c_d, r2_d, mae_d, _ = fit_ols(X_d, y)
        print(f"  D  food,pop,food²,terrain*(1-f):  R²={r2_d:.6f}")
        names_d = ["food", "pop", "food²", "pl*(1-f)", "fo*(1-f)", "mt*(1-f)", "oc*(1-f)"]
        for nm, c in zip(names_d, c_d):
            print(f"       {nm:>12s} = {c:>12.6f}")

        # ── Model E: food,food²,pop,terrain basic + terrain*(1-food) ───────
        X_e = dm[:, [0,1,2,3,4,5,6, 11,12,13,14]]
        c_e, r2_e, mae_e, _ = fit_ols(X_e, y)
        print(f"  E  food,food²,pop,terr+terr*(1-f):R²={r2_e:.6f}")

        # ── Model F: food,food²,pop,pop*food,terrain ───────────────────────
        X_f = dm[:, [0,1,2,3,4,5,6,9]]
        c_f, r2_f, mae_f, _ = fit_ols(X_f, y)
        print(f"  F  food,food²,pop,pop*food,terr:  R²={r2_f:.6f}")
        names_f = ["food", "pop", "pl", "fo", "mt", "oc", "food²", "pop*food"]
        for nm, c in zip(names_f, c_f):
            print(f"       {nm:>12s} = {c:>12.6f}")

        # ── Model G: Try food³ ─────────────────────────────────────────────
        food_cu = food ** 3
        food_cu_dm = demean_by_group(food_cu.reshape(-1, 1), groups).ravel()
        X_g = np.column_stack([dm[:, [0,1,2,3,4,5,6]], food_cu_dm])
        c_g, r2_g, _, _ = fit_ols(X_g, y)
        print(f"  G  + food³:                       R²={r2_g:.6f}  [food³={c_g[7]:.6f}]")

        # ── Model H: Try sqrt(food), log(food+eps) ────────────────────────
        sqrt_food = np.sqrt(food)
        sqrt_food_dm = demean_by_group(sqrt_food.reshape(-1, 1), groups).ravel()
        X_h = np.column_stack([dm[:, [1,2,3,4,5]], sqrt_food_dm])
        c_h, r2_h, _, _ = fit_ols(X_h, y)
        print(f"  H  sqrt(food),pop,terrain:        R²={r2_h:.6f}")

        log_food = np.log(food + 0.001)
        log_food_dm = demean_by_group(log_food.reshape(-1, 1), groups).ravel()
        X_i = np.column_stack([dm[:, [1,2,3,4,5]], log_food_dm])
        c_i, r2_i, _, _ = fit_ols(X_i, y)
        print(f"  I  log(food),pop,terrain:         R²={r2_i:.6f}")

        # ── Best model so far (B) — show residual patterns ─────────────────
        print(f"\n  Residual analysis of model B (food + food² + pop + terrain):")
        _, _, _, res_b = fit_ols(X_b, y)
        # Bin by food level
        food_bins = np.digitize(food, np.arange(0, 1.05, 0.1))
        print(f"    Residual by food level:")
        for b in range(1, 11):
            mask = food_bins == b
            if mask.sum() > 10:
                print(f"      food [{(b-1)*0.1:.1f}-{b*0.1:.1f}]: n={mask.sum():5d}, "
                      f"mean_res={res_b[mask].mean():>9.6f}, std={res_b[mask].std():.6f}")

        # Bin by pop level
        pop_bins = np.digitize(pop, np.arange(0, 3.0, 0.2))
        print(f"    Residual by pop level:")
        for b in range(1, 16):
            mask = pop_bins == b
            if mask.sum() > 10:
                print(f"      pop [{(b-1)*0.2:.1f}-{b*0.2:.1f}]: n={mask.sum():5d}, "
                      f"mean_res={res_b[mask].mean():>9.6f}, std={res_b[mask].std():.6f}")

        print()


if __name__ == "__main__":
    main()
