"""
Test hypothesis: food_delta = (terrain + feedback*food - pop_rate*pop + weather) * (1-food)

If true, then: food_delta / (1-food) = terrain + feedback*food - pop_rate*pop + weather
which is LINEAR. Demeaning removes weather.

This would explain why f*(1-f) coefficient varies — it's not a real term,
it's the weather×(1-f) interaction being misidentified as f*(1-f).

Reproduce:  python3 data-analysis/test_all_logistic.py
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


def fit_ols(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    res = y - pred
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def demean_col(vals, groups):
    group_ids = defaultdict(list)
    for i, g in enumerate(groups):
        group_ids[g].append(i)
    result = vals.copy()
    for indices in group_ids.values():
        result[indices] -= vals[indices].mean()
    return result


def collect_stable_rows(replays):
    rows = []
    groups = []
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
                rows.append((
                    sb["population"], sb["food"],
                    adj_before.get(PLAINS, 0), adj_before.get(FOREST, 0),
                    adj_before.get(MOUNTAIN, 0), adj_before.get(OCEAN, 0),
                    adj_before.get(SETTLEMENT, 0) + adj_before.get(PORT, 0) + adj_before.get(RUIN, 0),
                    sa["food"] - sb["food"],
                ))
                groups.append((seed, step))
    return np.array(rows, dtype=np.float64), groups


def iterative_clean(arr, groups, y_col_fn, feature_fn, n_iters=3, threshold=3.0):
    for _ in range(n_iters):
        X = feature_fn(arr)
        y = y_col_fn(arr)
        dm_X = np.zeros_like(X)
        dm_y = demean_col(y, groups)
        for col in range(X.shape[1]):
            dm_X[:, col] = demean_col(X[:, col], groups)
        _, _, mae, res = fit_ols(dm_X, dm_y)
        inlier = np.abs(res) < threshold * mae
        arr = arr[inlier]
        groups = [groups[i] for i in range(len(groups)) if inlier[i]]
    X = feature_fn(arr)
    y = y_col_fn(arr)
    dm_X = np.zeros_like(X)
    dm_y = demean_col(y, groups)
    for col in range(X.shape[1]):
        dm_X[:, col] = demean_col(X[:, col], groups)
    c, r2, mae, res = fit_ols(dm_X, dm_y)
    return c, r2, mae, res, arr, groups


def main():
    rounds = load_replays_by_round()

    print("=" * 80)
    print("HYPOTHESIS: food_delta = (terrain + feedback*food - pop_cost*pop + weather) × (1-food)")
    print("  => food_delta/(1-food) = terrain + feedback*food - pop_cost*pop + weather")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════
    # Model A: food_delta = terrain*(1-f) + fb*f*(1-f) - pop*pop + weather  [current best]
    # Model B: food_delta/(1-f) = terrain + fb*food - pop*pop + weather     [all-logistic]
    # Model C: food_delta = terrain*(1-f) - pop*pop + weather               [no feedback]
    # Model D: food_delta/(1-f) = terrain - pop*pop + weather               [all-logistic, no fb]
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'Round':<15} {'A: curr':>8} {'B: all-log':>10} {'C: no-fb':>10} {'D: allog-nofb':>14}")

    all_coeff_A = []
    all_coeff_B = []

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)

        # Skip observations where food is very close to 1 (division by near-zero)
        food = arr[:, 1]
        safe = food < 0.95  # avoid division issues
        arr_safe = arr[safe]
        groups_safe = [groups[i] for i in range(len(groups)) if safe[i]]

        # Model A: current best
        def feat_A(a):
            omf = 1 - a[:, 1]
            return np.column_stack([a[:, 2]*omf, a[:, 3]*omf, a[:, 1]*omf, a[:, 0]])
        def y_A(a):
            return a[:, 7]

        c_A, r2_A, _, _, _, _ = iterative_clean(arr.copy(), list(groups), y_A, feat_A)
        all_coeff_A.append(c_A)

        # Model B: all-logistic (divide by (1-f))
        def feat_B(a):
            return np.column_stack([a[:, 2], a[:, 3], a[:, 1], a[:, 0]])
        def y_B(a):
            omf = 1 - a[:, 1]
            return a[:, 7] / omf

        c_B, r2_B, _, _, _, _ = iterative_clean(arr_safe.copy(), list(groups_safe), y_B, feat_B)
        all_coeff_B.append(c_B)

        # Model C: no feedback
        def feat_C(a):
            omf = 1 - a[:, 1]
            return np.column_stack([a[:, 2]*omf, a[:, 3]*omf, a[:, 0]])
        c_C, r2_C, _, _, _, _ = iterative_clean(arr.copy(), list(groups), y_A, feat_C)

        # Model D: all-logistic, no feedback
        def feat_D(a):
            return np.column_stack([a[:, 2], a[:, 3], a[:, 0]])
        c_D, r2_D, _, _, _, _ = iterative_clean(arr_safe.copy(), list(groups_safe), y_B, feat_D)

        print(f"  {rid[:14]:<15} {r2_A:>8.4f} {r2_B:>10.4f} {r2_C:>10.4f} {r2_D:>14.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("COEFFICIENT COMPARISON")
    print(f"{'='*80}")

    print("\nModel A: food_delta = pl*(1-f) + fo*(1-f) + fb*f*(1-f) - pop + weather")
    all_A = np.array(all_coeff_A)
    print(f"  pl*(1-f): {all_A[:,0].mean():.6f} ± {all_A[:,0].std():.6f} (CV={all_A[:,0].std()/all_A[:,0].mean()*100:.1f}%)")
    print(f"  fo*(1-f): {all_A[:,1].mean():.6f} ± {all_A[:,1].std():.6f} (CV={all_A[:,1].std()/all_A[:,1].mean()*100:.1f}%)")
    print(f"  f*(1-f):  {all_A[:,2].mean():.6f} ± {all_A[:,2].std():.6f} (CV={all_A[:,2].std()/all_A[:,2].mean()*100:.1f}%)")
    print(f"  pop:      {all_A[:,3].mean():.6f} ± {all_A[:,3].std():.6f} (CV={all_A[:,3].std()/all_A[:,3].mean()*100:.1f}%)")

    print("\nModel B: food_delta/(1-f) = pl + fo + fb*food - pop + weather")
    all_B = np.array(all_coeff_B)
    print(f"  pl:   {all_B[:,0].mean():.6f} ± {all_B[:,0].std():.6f} (CV={all_B[:,0].std()/all_B[:,0].mean()*100:.1f}%)")
    print(f"  fo:   {all_B[:,1].mean():.6f} ± {all_B[:,1].std():.6f} (CV={all_B[:,1].std()/all_B[:,1].mean()*100:.1f}%)")
    print(f"  food: {all_B[:,2].mean():.6f} ± {all_B[:,2].std():.6f} (CV={all_B[:,2].std()/all_B[:,2].mean()*100:.1f}%)")
    print(f"  pop:  {all_B[:,3].mean():.6f} ± {all_B[:,3].std():.6f} (CV={all_B[:,3].std()/all_B[:,3].mean()*100:.1f}%)")

    ratios_A = all_A[:, 1] / all_A[:, 0]
    ratios_B = all_B[:, 1] / all_B[:, 0]
    print(f"\n  fo/pl ratios:")
    print(f"    Model A: {ratios_A.mean():.3f} ± {ratios_A.std():.3f}")
    print(f"    Model B: {ratios_B.mean():.3f} ± {ratios_B.std():.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Test: everything × (1-f), including pop
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("EXTRA: What if pop also scales with (1-f)?")
    print("  food_delta = (terrain + fb*food - pop*pop) × (1-food)")
    print(f"{'='*80}")

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        food = arr[:, 1]
        safe = food < 0.95
        arr_safe = arr[safe]
        groups_safe = [groups[i] for i in range(len(groups)) if safe[i]]

        # Model B2: food_delta/(1-f) = terrain + fb*food - pop_rate*pop
        def feat_B2(a):
            return np.column_stack([a[:, 2], a[:, 3], a[:, 1], a[:, 0]])
        def y_B2(a):
            return a[:, 7] / (1 - a[:, 1])

        c, r2, mae, _, arr_c, _ = iterative_clean(arr_safe.copy(), list(groups_safe), y_B2, feat_B2)

        # Verify: reconstruct food_delta and check against observed
        omf_c = 1 - arr_c[:, 1]
        pred = (c[0] * arr_c[:, 2] + c[1] * arr_c[:, 3] + c[2] * arr_c[:, 1] + c[3] * arr_c[:, 0]) * omf_c
        # We need to add back weather. Compute group means.
        actual = arr_c[:, 7]
        residual = actual - pred
        groups_c = [groups_safe[i] for i in range(len(groups_safe)) if i < len(arr_c)]  # approximation

        print(f"\n  {rid[:12]}: R²={r2:.6f} MAE={mae:.6f}")
        print(f"    pl={c[0]:.6f} fo={c[1]:.6f} food={c[2]:.6f} pop={c[3]:.6f}")
        print(f"    fo/pl={c[1]/c[0]:.4f}")


if __name__ == "__main__":
    main()
