"""
Final food formula derivation.

Combine insights:
1. Terrain-stable data only (filter terrain changes between frames)
2. Model: terrain*(1-food) + food*(1-food) + pop + pop*food
3. Iterative outlier removal (3x MAE threshold)
4. Per-round fitting to get exact coefficients

Also test: whether terrain coefficients are truly constant across rounds
or proportional.

Reproduce:  python3 data-analysis/final_food_formula.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def fit_and_clean(arr, groups, n_iters=3, threshold_mult=3.0, model="base", verbose=True):
    """Fit model with iterative outlier removal. Returns final coefficients and R²."""
    pop = arr[:, 0]
    food = arr[:, 1]
    n_pl, n_fo = arr[:, 2], arr[:, 3]
    food_delta = arr[:, 7]
    omf = 1 - food

    for iteration in range(n_iters + 1):
        if model == "base":
            features = np.column_stack([
                n_pl * omf, n_fo * omf, food * omf, pop, food_delta
            ])
            n_feat = 4
        elif model == "pop_food":
            features = np.column_stack([
                n_pl * omf, n_fo * omf, food * omf, pop, pop * food, food_delta
            ])
            n_feat = 5
        elif model == "linear":
            features = np.column_stack([
                n_pl, n_fo, food, pop, food_delta
            ])
            n_feat = 4
        elif model == "linear_pop_food":
            features = np.column_stack([
                n_pl, n_fo, food, pop, pop * food, food_delta
            ])
            n_feat = 5
        else:
            raise ValueError(f"Unknown model: {model}")

        dm = np.zeros_like(features)
        for col in range(features.shape[1]):
            dm[:, col] = demean_col(features[:, col], groups)

        c, r2, mae, res = fit_ols(dm[:, :n_feat], dm[:, n_feat])

        if verbose and iteration == 0:
            print(f"  Initial: n={len(arr)}, R²={r2:.6f}, MAE={mae:.6f}")

        if iteration < n_iters:
            inlier = np.abs(res) < threshold_mult * mae
            n_removed = (~inlier).sum()
            if verbose:
                print(f"  Iter {iteration+1}: removed {n_removed} ({100*n_removed/len(arr):.1f}%)")
            arr = arr[inlier]
            groups = [groups[i] for i in range(len(groups)) if inlier[i]]
            pop = arr[:, 0]
            food = arr[:, 1]
            n_pl, n_fo = arr[:, 2], arr[:, 3]
            food_delta = arr[:, 7]
            omf = 1 - food

    if verbose:
        print(f"  Final: n={len(arr)}, R²={r2:.6f}, MAE={mae:.6f}")

    return c, r2, mae, res, arr, groups


def main():
    rounds = load_replays_by_round()

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Compare models per-round with outlier removal
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PART 1: Per-round model comparison (terrain-stable, 3 rounds outlier removal)")
    print("=" * 80)

    results = {}

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        print(f"\n{'─'*72}")
        print(f"Round {rid[:12]} ({len(arr)} terrain-stable obs)")

        print("\n  [Base model: pl*(1-f) + fo*(1-f) + f*(1-f) + pop]")
        c_base, r2_base, mae_base, _, arr_clean, _ = fit_and_clean(
            arr.copy(), list(groups), model="base")

        print("\n  [Extended model: + pop*food]")
        c_ext, r2_ext, mae_ext, _, arr_clean_ext, _ = fit_and_clean(
            arr.copy(), list(groups), model="pop_food")

        print("\n  [Linear model: pl + fo + food + pop (no (1-f) interactions)]")
        c_lin, r2_lin, mae_lin, _, _, _ = fit_and_clean(
            arr.copy(), list(groups), model="linear")

        results[rid] = {
            "base": (c_base, r2_base),
            "ext": (c_ext, r2_ext),
            "linear": (c_lin, r2_lin),
        }

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Summary table
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 2: Summary — final R² after outlier removal")
    print(f"{'='*80}")
    print(f"{'Round':<15} {'Base':>8} {'+pop*f':>8} {'Linear':>8}")
    for rid in sorted(rounds):
        r = results[rid]
        print(f"{rid[:14]:<15} {r['base'][1]:>8.4f} {r['ext'][1]:>8.4f} {r['linear'][1]:>8.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Coefficient table for best model
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 3: Coefficients for base model (after outlier removal)")
    print("  food_delta = c0*pl*(1-f) + c1*fo*(1-f) + c2*f*(1-f) + c3*pop + weather")
    print(f"{'='*80}")
    print(f"{'Round':<15} {'pl*(1-f)':>10} {'fo*(1-f)':>10} {'f*(1-f)':>10} {'pop':>10} {'fo/pl':>8}")

    all_c = []
    for rid in sorted(rounds):
        c = results[rid]["base"][0]
        ratio = c[1] / c[0] if abs(c[0]) > 1e-8 else float('nan')
        print(f"{rid[:14]:<15} {c[0]:>10.6f} {c[1]:>10.6f} {c[2]:>10.6f} {c[3]:>10.6f} {ratio:>8.3f}")
        all_c.append(c)

    all_c = np.array(all_c)
    means = all_c.mean(axis=0)
    stds = all_c.std(axis=0)
    ratio_mean = means[1] / means[0]
    print(f"{'MEAN':<15} {means[0]:>10.6f} {means[1]:>10.6f} {means[2]:>10.6f} {means[3]:>10.6f} {ratio_mean:>8.3f}")
    print(f"{'STD':<15} {stds[0]:>10.6f} {stds[1]:>10.6f} {stds[2]:>10.6f} {stds[3]:>10.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: Coefficients for extended model
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 4: Coefficients for extended model (after outlier removal)")
    print("  food_delta = c0*pl*(1-f) + c1*fo*(1-f) + c2*f*(1-f) + c3*pop + c4*pop*f + weather")
    print(f"{'='*80}")
    print(f"{'Round':<15} {'pl*(1-f)':>10} {'fo*(1-f)':>10} {'f*(1-f)':>10} {'pop':>10} {'pop*f':>10}")

    all_c_ext = []
    for rid in sorted(rounds):
        c = results[rid]["ext"][0]
        print(f"{rid[:14]:<15} {c[0]:>10.6f} {c[1]:>10.6f} {c[2]:>10.6f} {c[3]:>10.6f} {c[4]:>10.6f}")
        all_c_ext.append(c)

    all_c_ext = np.array(all_c_ext)
    means_ext = all_c_ext.mean(axis=0)
    stds_ext = all_c_ext.std(axis=0)
    print(f"{'MEAN':<15} {means_ext[0]:>10.6f} {means_ext[1]:>10.6f} {means_ext[2]:>10.6f} {means_ext[3]:>10.6f} {means_ext[4]:>10.6f}")
    print(f"{'STD':<15} {stds_ext[0]:>10.6f} {stds_ext[1]:>10.6f} {stds_ext[2]:>10.6f} {stds_ext[3]:>10.6f} {stds_ext[4]:>10.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: Are terrain coefficients constant across rounds?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 5: Testing if terrain coefficients are constant")
    print(f"{'='*80}")

    print(f"\n  pl*(1-f): mean={means[0]:.6f} ± {stds[0]:.6f} (CV={stds[0]/means[0]*100:.1f}%)")
    print(f"  fo*(1-f): mean={means[1]:.6f} ± {stds[1]:.6f} (CV={stds[1]/means[1]*100:.1f}%)")
    print(f"  f*(1-f):  mean={means[2]:.6f} ± {stds[2]:.6f} (CV={stds[2]/means[2]*100:.1f}%)")
    print(f"  pop:      mean={means[3]:.6f} ± {stds[3]:.6f} (CV={stds[3]/means[3]*100:.1f}%)")

    print(f"\n  fo/pl ratio: {ratio_mean:.4f}")
    ratios = all_c[:, 1] / all_c[:, 0]
    print(f"  per-round ratios: {', '.join(f'{r:.3f}' for r in ratios)}")
    print(f"  ratio std: {ratios.std():.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 6: Residual visualization for best model
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)

        pop = arr[:, 0]
        food = arr[:, 1]
        n_pl, n_fo = arr[:, 2], arr[:, 3]
        food_delta = arr[:, 7]
        omf = 1 - food

        features = np.column_stack([
            n_pl * omf, n_fo * omf, food * omf, pop, food_delta
        ])
        dm = np.zeros_like(features)
        for col in range(features.shape[1]):
            dm[:, col] = demean_col(features[:, col], groups)

        c, r2, mae, res = fit_ols(dm[:, :4], dm[:, 4])

        ax = axes[idx]
        ax.scatter(food, res, s=1, alpha=0.1, c='blue')

        # Plot median curve
        bins = np.linspace(0, 1, 51)
        centers = (bins[:-1] + bins[1:]) / 2
        medians = []
        for j in range(len(bins) - 1):
            bm = (food >= bins[j]) & (food < bins[j+1])
            if bm.sum() >= 10:
                medians.append(np.median(res[bm]))
            else:
                medians.append(np.nan)
        ax.plot(centers, medians, 'r-', lw=2, label='median')
        ax.axhline(y=0, color='gray', lw=0.5)
        ax.set_xlabel('food')
        ax.set_ylabel('residual')
        ax.set_title(f'{rid[:8]} R²={r2:.4f}')
        ax.set_ylim(-0.2, 0.2)
        ax.legend(fontsize=7)

    fig.suptitle('Base model residual vs food (terrain-stable, before outlier removal)', fontsize=14)
    plt.tight_layout()
    fig.savefig('final_residual_vs_food.png', dpi=150)
    print(f"\nSaved final_residual_vs_food.png")
    plt.close('all')


if __name__ == "__main__":
    main()
