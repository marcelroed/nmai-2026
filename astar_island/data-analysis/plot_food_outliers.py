"""
Find the clean formula by identifying + removing outliers.

Approach:
1. Fit the model on all non-raided, non-port data
2. Look at residual distribution — is it bimodal?
3. Remove outliers, re-fit, check if R² → 1.0
4. Characterize what outliers have in common

Reproduce:  python3 data-analysis/plot_food_outliers.py
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


def collect_rows(replays):
    """Collect ALL data (except ports) — don't filter anything else."""
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
                x, y = pos
                if grid[y][x] == PORT:
                    continue
                # Keep EVERYTHING — including capped, raided, growth
                adj = terrain_adj_full(grid, x, y)
                rows.append((
                    sb["population"], sb["food"],
                    adj.get(PLAINS, 0), adj.get(FOREST, 0),
                    adj.get(MOUNTAIN, 0), adj.get(OCEAN, 0),
                    adj.get(SETTLEMENT, 0) + adj.get(PORT, 0) + adj.get(RUIN, 0),
                    sa["food"] - sb["food"],
                    sa["defense"] - sb["defense"],
                    sa["population"] - sb["population"],
                    sa["food"],
                    sb["defense"],
                    sb["wealth"],
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
    group_ids = defaultdict(list)
    for i, g in enumerate(groups):
        group_ids[g].append(i)
    result = arr.copy()
    for indices in group_ids.values():
        result[indices] -= vals_mean(arr, indices)
    return result


def vals_mean(arr, indices):
    if arr.ndim == 1:
        return arr[indices].mean()
    return arr[indices].mean(axis=0)


def demean_col(vals, groups):
    group_ids = defaultdict(list)
    for i, g in enumerate(groups):
        group_ids[g].append(i)
    result = vals.copy()
    for indices in group_ids.values():
        result[indices] -= vals[indices].mean()
    return result


def main():
    rounds = load_replays_by_round()

    fig_res, axes_res = plt.subplots(3, 3, figsize=(20, 18))
    fig_scatter, axes_scatter = plt.subplots(3, 3, figsize=(20, 18))
    axes_res = axes_res.flatten()
    axes_scatter = axes_scatter.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        replays = rounds[rid]
        arr, groups = collect_rows(replays)
        n = len(arr)

        pop = arr[:, 0]
        food = arr[:, 1]
        n_pl, n_fo, n_mt, n_oc, n_st = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
        food_delta = arr[:, 7]
        def_delta = arr[:, 8]
        pop_delta = arr[:, 9]
        food_after = arr[:, 10]
        def_before = arr[:, 11]
        wealth = arr[:, 12]
        omf = 1 - food

        # Classify each observation
        raided = def_delta < -0.001
        capped_high = food_after >= 0.997
        capped_low = food_after <= 0.001
        grew = pop_delta < -0.05

        # "Clean" = not raided, not capped, not grew
        clean = (~raided) & (~capped_high) & (~capped_low) & (~grew)

        print(f"\n{'='*72}")
        print(f"Round {rid[:12]}: {n} total observations")
        print(f"  raided: {raided.sum()} ({100*raided.mean():.1f}%)")
        print(f"  capped high: {capped_high.sum()} ({100*capped_high.mean():.1f}%)")
        print(f"  capped low: {capped_low.sum()} ({100*capped_low.mean():.1f}%)")
        print(f"  grew: {grew.sum()} ({100*grew.mean():.1f}%)")
        print(f"  clean: {clean.sum()} ({100*clean.mean():.1f}%)")

        # Fit model on clean data only
        food_c = food[clean]
        pop_c = pop[clean]
        omf_c = omf[clean]
        fd_c = food_delta[clean]
        npl_c = n_pl[clean]
        nfo_c = n_fo[clean]
        nst_c = n_st[clean]
        groups_c = [groups[i] for i in range(n) if clean[i]]

        features = np.column_stack([
            npl_c * omf_c, nfo_c * omf_c,
            food_c * omf_c,  # food*(1-food)
            pop_c,
            fd_c,
        ])
        # Demean each column by group
        dm = np.zeros_like(features)
        for col in range(features.shape[1]):
            dm[:, col] = demean_col(features[:, col], groups_c)

        c, r2, mae, res = fit_ols(dm[:, :4], dm[:, 4])
        print(f"  Clean fit: R²={r2:.6f}, MAE={mae:.6f}")
        print(f"    pl*(1-f)={c[0]:.6f}, fo*(1-f)={c[1]:.6f}, "
              f"f*(1-f)={c[2]:.6f}, pop={c[3]:.6f}")

        # ── Iterative outlier removal ────────────────────────────────────
        threshold_mult = 3.0  # remove points > 3*MAE
        for iteration in range(3):
            inlier = np.abs(res) < threshold_mult * mae
            n_removed = (~inlier).sum()
            pct_removed = 100 * n_removed / len(res)

            # Re-index to inlier subset
            food_ci = food_c[inlier]
            pop_ci = pop_c[inlier]
            omf_ci = 1 - food_ci
            fd_ci = fd_c[inlier]
            npl_ci = npl_c[inlier]
            nfo_ci = nfo_c[inlier]
            groups_ci = [groups_c[i] for i in range(len(groups_c)) if inlier[i]]

            features_i = np.column_stack([
                npl_ci * omf_ci, nfo_ci * omf_ci,
                food_ci * omf_ci,
                pop_ci,
                fd_ci,
            ])
            dm_i = np.zeros_like(features_i)
            for col in range(features_i.shape[1]):
                dm_i[:, col] = demean_col(features_i[:, col], groups_ci)

            c_i, r2_i, mae_i, res_i = fit_ols(dm_i[:, :4], dm_i[:, 4])
            print(f"  Iter {iteration+1} (removed {n_removed}/{len(res)} = {pct_removed:.1f}%): "
                  f"R²={r2_i:.6f}, MAE={mae_i:.6f}")
            print(f"    pl*(1-f)={c_i[0]:.6f}, fo*(1-f)={c_i[1]:.6f}, "
                  f"f*(1-f)={c_i[2]:.6f}, pop={c_i[3]:.6f}")

            # Update for next iteration
            food_c = food_ci
            pop_c = pop_ci
            omf_c = omf_ci
            fd_c = fd_ci
            npl_c = npl_ci
            nfo_c = nfo_ci
            groups_c = groups_ci
            res = res_i
            mae = mae_i

        # ── Histogram of initial residuals ───────────────────────────────
        ax = axes_res[idx]
        # Re-compute initial residuals for histogram
        food_c0 = food[clean]
        pop_c0 = pop[clean]
        omf_c0 = omf[clean]
        fd_c0 = food_delta[clean]
        npl_c0 = n_pl[clean]
        nfo_c0 = n_fo[clean]
        groups_c0 = [groups[i] for i in range(n) if clean[i]]

        features0 = np.column_stack([
            npl_c0 * omf_c0, nfo_c0 * omf_c0,
            food_c0 * omf_c0,
            pop_c0,
            fd_c0,
        ])
        dm0 = np.zeros_like(features0)
        for col in range(features0.shape[1]):
            dm0[:, col] = demean_col(features0[:, col], groups_c0)
        _, _, _, res0 = fit_ols(dm0[:, :4], dm0[:, 4])

        ax.hist(res0, bins=200, range=(-0.3, 0.3), density=True, alpha=0.7)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_title(f'Round {rid[:8]} (clean only)')
        ax.set_xlabel('residual')
        ax.set_ylabel('density')

        # ── Scatter: residual vs pop_delta ───────────────────────────────
        ax2 = axes_scatter[idx]
        pop_delta_c0 = pop_delta[clean]
        ax2.scatter(pop_delta_c0, res0, s=1, alpha=0.2)
        ax2.set_xlabel('pop_delta')
        ax2.set_ylabel('residual')
        ax2.set_title(f'Round {rid[:8]}: residual vs pop_delta')
        ax2.axhline(y=0, color='gray', linewidth=0.5)
        ax2.axvline(x=0, color='gray', linewidth=0.5)
        ax2.set_xlim(-0.5, 1.0)
        ax2.set_ylim(-0.3, 0.3)

    fig_res.suptitle('Residual distribution (clean observations, terrain*(1-f)+f*(1-f)+pop model)',
                     fontsize=14)
    plt.figure(fig_res)
    plt.tight_layout()
    fig_res.savefig('food_residual_clean.png', dpi=150)
    print("\nSaved food_residual_clean.png")

    fig_scatter.suptitle('Residual vs pop_delta — looking for missed growth events', fontsize=14)
    plt.figure(fig_scatter)
    plt.tight_layout()
    fig_scatter.savefig('food_residual_vs_popdelta.png', dpi=150)
    print("Saved food_residual_vs_popdelta.png")
    plt.close('all')


if __name__ == "__main__":
    main()
