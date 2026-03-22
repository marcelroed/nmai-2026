"""
Find the food production formula using fixed-effects regression.

Reproduce:  python3 data-analysis/analyze_food_production.py

Key insight: there is a per-(seed, step) stochastic component (weather) that
is the SAME for all settlements in a given step.  By demeaning within each
(seed, step) group, the weather cancels out and we can estimate the
deterministic coefficients (pop, food, terrain) cleanly.

Model:
  food_delta_{i,t} = f(pop, food, terrain) + weather_t

  Demeaning within step t removes weather_t:
    food_delta_{i,t} - mean_t = f(...) - mean_t(f(...))
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
    """Collect (pop_b, food_b, n_pl, n_fo, n_mt, n_oc, food_delta, group_id)
    for non-port, non-raided settlements. group_id = (seed_index, step)."""
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
                    continue  # skip ports (trade food)
                # Skip capped food
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
    """Subtract group means from each column."""
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

        pop_b = arr[:, 0]
        food_b = arr[:, 1]
        n_pl, n_fo, n_mt, n_oc = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        food_delta = arr[:, 6]

        n_groups = len(set(groups))

        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}  ({n} obs, {n_groups} (seed,step) groups)")
        print(f"{'=' * 72}")

        # ── A. Naive (no fixed effects) ────────────────────────────────────
        X_naive = np.column_stack([
            np.ones(n), food_b, pop_b, n_pl, n_fo, n_mt, n_oc,
        ])
        c_n, r2_n, mae_n, _ = fit_ols(X_naive, food_delta)
        print(f"  Naive:        R²={r2_n:.6f}, MAE={mae_n:.6f}")

        # ── B. Fixed effects (demean by group) ─────────────────────────────
        features = np.column_stack([food_b, pop_b, n_pl, n_fo, n_mt, n_oc, food_delta])
        features_dm = demean_by_group(features, groups)

        X_fe = features_dm[:, :6]  # demeaned features (no intercept needed)
        y_fe = features_dm[:, 6]   # demeaned food_delta

        c_fe, r2_fe, mae_fe, res_fe = fit_ols(X_fe, y_fe)
        print(f"  Fixed-effects: R²={r2_fe:.6f}, MAE={mae_fe:.6f}")
        names = ["food_b", "pop_b", "n_plains", "n_forest", "n_mountain", "n_ocean"]
        for name, c in zip(names, c_fe):
            print(f"    {name:>12s} = {c:>12.8f}")

        # ── C. Stability across seeds ──────────────────────────────────────
        print(f"\n  Per-seed fixed-effects coefficients:")
        for replay in replays:
            seed_rows, seed_groups = collect_rows([replay])
            if len(seed_rows) < 50:
                continue
            feats = np.column_stack([
                seed_rows[:, 1], seed_rows[:, 0],
                seed_rows[:, 2], seed_rows[:, 3], seed_rows[:, 4], seed_rows[:, 5],
                seed_rows[:, 6],
            ])
            feats_dm = demean_by_group(feats, seed_groups)
            c_s, r2_s, mae_s, _ = fit_ols(feats_dm[:, :6], feats_dm[:, 6])
            coeffs_str = "  ".join(f"{n}={c:.6f}" for n, c in zip(names, c_s))
            print(f"    seed={replay['seed_index']}: R²={r2_s:.6f}  {coeffs_str}")

        # ── D. Check residuals: try food², pop², interactions ──────────────
        print(f"\n  Residual analysis (after fixed-effects):")
        # Try adding food²
        food_b_dm = features_dm[:, 0]
        pop_b_dm = features_dm[:, 1]
        food_sq_dm = demean_by_group(
            (food_b ** 2).reshape(-1, 1), groups
        ).ravel()

        X_sq = np.column_stack([X_fe, food_sq_dm])
        c_sq, r2_sq, mae_sq, _ = fit_ols(X_sq, y_fe)
        print(f"    + food²:    R²={r2_sq:.6f}  [food²={c_sq[6]:.6f}]")

        # Try pop*food
        pf_dm = demean_by_group(
            (pop_b * food_b).reshape(-1, 1), groups
        ).ravel()
        X_pf = np.column_stack([X_fe, pf_dm])
        c_pf, r2_pf, mae_pf, _ = fit_ols(X_pf, y_fe)
        print(f"    + pop*food:  R²={r2_pf:.6f}  [pop*food={c_pf[6]:.6f}]")

        # Try pop²
        pop_sq_dm = demean_by_group(
            (pop_b ** 2).reshape(-1, 1), groups
        ).ravel()
        X_p2 = np.column_stack([X_fe, pop_sq_dm])
        c_p2, r2_p2, mae_p2, _ = fit_ols(X_p2, y_fe)
        print(f"    + pop²:      R²={r2_p2:.6f}  [pop²={c_p2[6]:.6f}]")

        # ── E. What fraction of total variance is weather? ─────────────────
        total_var = np.var(food_delta)
        within_var = np.var(y_fe)
        between_var = total_var - within_var
        print(f"\n  Variance decomposition:")
        print(f"    total var:   {total_var:.8f}")
        print(f"    between (weather): {between_var:.8f} ({100*between_var/total_var:.1f}%)")
        print(f"    within (deterministic): {within_var:.8f} ({100*within_var/total_var:.1f}%)")
        print(f"    FE model explains {100*r2_fe*within_var/total_var:.1f}% of total variance")

        print()


if __name__ == "__main__":
    main()
