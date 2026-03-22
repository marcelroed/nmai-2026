"""
Plot food production formula proof.

Shows ALL data points with outliers colored by reason, and the fitted curve
for the clean (non-outlier) data overlaid.

Since food_delta depends on multiple variables (terrain, food, pop, weather),
we plot the DEMEANED food_delta (weather removed) against the model's
predicted value. This collapses the multi-dimensional relationship into
a single predicted-vs-actual scatter.

Reproduce:  python3 data-analysis/plot_food_proof.py
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


def terrain_adj(grid, x, y):
    h, w = len(grid), len(grid[0])
    counts = defaultdict(int)
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if nx == x and ny == y:
                continue
            counts[grid[ny][nx]] += 1
    return counts


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
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, res


def demean_col(vals, groups):
    gmap = defaultdict(list)
    for i, g in enumerate(groups):
        gmap[g].append(i)
    out = vals.copy()
    for indices in gmap.values():
        out[indices] -= vals[indices].mean()
    return out


def collect_all_classified(replays):
    """Collect ALL observations and classify each one."""
    rows = []
    for replay in replays:
        frames = replay["frames"]
        seed = replay["seed_index"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            grid_b = fb["grid"]
            grid_a = fa["grid"]
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

                adj_b = terrain_adj(grid_b, x, y)
                adj_a = terrain_adj(grid_a, x, y)

                is_port = grid_b[y][x] == PORT
                is_raided = (sa["defense"] - sb["defense"]) < -0.001
                is_capped_hi = sa["food"] >= 0.997
                is_capped_lo = sa["food"] <= 0.001
                pop_delta = sa["population"] - sb["population"]
                is_grew = pop_delta < -0.05
                terrain_changed = dict(adj_b) != dict(adj_a)

                # Classify: pick the FIRST applicable reason
                if is_port:
                    reason = "port"
                elif is_raided:
                    reason = "raided"
                elif is_capped_hi or is_capped_lo:
                    reason = "capped"
                elif is_grew:
                    reason = "growth"
                elif terrain_changed:
                    reason = "terrain_changed"
                else:
                    reason = "clean"

                rows.append({
                    "pop": sb["population"],
                    "food": sb["food"],
                    "n_pl": adj_b.get(PLAINS, 0),
                    "n_fo": adj_b.get(FOREST, 0),
                    "food_delta": sa["food"] - sb["food"],
                    "seed": seed,
                    "step": step,
                    "reason": reason,
                })

    return rows


def main():
    rounds = load_replays_by_round()

    # Colors and labels for each reason
    reason_config = {
        "clean":           {"color": "#2196F3", "label": "Clean",           "zorder": 2, "s": 3,  "alpha": 0.15},
        "terrain_changed": {"color": "#FF9800", "label": "Terrain changed", "zorder": 3, "s": 5,  "alpha": 0.3},
        "growth":          {"color": "#E91E63", "label": "Growth event",    "zorder": 4, "s": 8,  "alpha": 0.5},
        "raided":          {"color": "#F44336", "label": "Raided",          "zorder": 4, "s": 8,  "alpha": 0.5},
        "capped":          {"color": "#9C27B0", "label": "Capped (0/1)",    "zorder": 4, "s": 8,  "alpha": 0.5},
        "port":            {"color": "#795548", "label": "Port",            "zorder": 3, "s": 5,  "alpha": 0.4},
    }

    n_rounds = len(rounds)
    cols = 3
    rows_grid = (n_rounds + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(7 * cols, 6 * rows_grid))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]
        all_obs = collect_all_classified(replays)

        # ── Step 1: fit model on clean data only ──────────────────────
        clean_obs = [o for o in all_obs if o["reason"] == "clean"]
        n_clean = len(clean_obs)

        pop_c = np.array([o["pop"] for o in clean_obs])
        food_c = np.array([o["food"] for o in clean_obs])
        npl_c = np.array([o["n_pl"] for o in clean_obs], dtype=float)
        nfo_c = np.array([o["n_fo"] for o in clean_obs], dtype=float)
        fd_c = np.array([o["food_delta"] for o in clean_obs])
        omf_c = 1 - food_c
        groups_c = [(o["seed"], o["step"]) for o in clean_obs]

        feat_cols = [npl_c * omf_c, nfo_c * omf_c, food_c * omf_c, pop_c]
        dm_X = np.column_stack([demean_col(c, groups_c) for c in feat_cols])
        dm_y = demean_col(fd_c, groups_c)

        coeffs, r2_raw, res_raw = fit_ols(dm_X, dm_y)

        # Iterative outlier removal to get clean coefficients
        mask_clean = np.ones(n_clean, dtype=bool)
        arr_work = np.column_stack([pop_c, food_c, npl_c, nfo_c, fd_c])
        grp_work = list(groups_c)
        for _ in range(3):
            p, f = arr_work[:, 0], arr_work[:, 1]
            np_, nf_ = arr_work[:, 2], arr_work[:, 3]
            fd_ = arr_work[:, 4]
            of_ = 1 - f
            fcols = [np_ * of_, nf_ * of_, f * of_, p]
            dmX = np.column_stack([demean_col(c, grp_work) for c in fcols])
            dmy = demean_col(fd_, grp_work)
            c_iter, _, res_iter = fit_ols(dmX, dmy)
            mae = np.mean(np.abs(res_iter))
            keep = np.abs(res_iter) < 3 * mae
            arr_work = arr_work[keep]
            grp_work = [grp_work[i] for i in range(len(grp_work)) if keep[i]]

        # Final fit on cleaned data
        p, f = arr_work[:, 0], arr_work[:, 1]
        np_, nf_ = arr_work[:, 2], arr_work[:, 3]
        fd_ = arr_work[:, 4]
        of_ = 1 - f
        fcols = [np_ * of_, nf_ * of_, f * of_, p]
        dmX = np.column_stack([demean_col(c, grp_work) for c in fcols])
        dmy = demean_col(fd_, grp_work)
        coeffs_final, r2_final, _ = fit_ols(dmX, dmy)

        # Also mark which clean observations became statistical outliers
        # Recompute residuals on ALL clean data using final coefficients
        dm_X_all = np.column_stack([demean_col(c, groups_c) for c in feat_cols])
        dm_y_all = demean_col(fd_c, groups_c)
        pred_all = dm_X_all @ coeffs_final
        res_all = dm_y_all - pred_all
        mae_final = np.mean(np.abs(res_iter))  # use clean MAE as threshold
        is_stat_outlier = np.abs(res_all) > 3 * mae_final

        # ── Step 2: compute predicted value for ALL observations ──────
        all_pop = np.array([o["pop"] for o in all_obs])
        all_food = np.array([o["food"] for o in all_obs])
        all_npl = np.array([o["n_pl"] for o in all_obs], dtype=float)
        all_nfo = np.array([o["n_fo"] for o in all_obs], dtype=float)
        all_fd = np.array([o["food_delta"] for o in all_obs])
        all_omf = 1 - all_food
        all_groups = [(o["seed"], o["step"]) for o in all_obs]
        all_reasons = [o["reason"] for o in all_obs]

        all_feat = [all_npl * all_omf, all_nfo * all_omf, all_food * all_omf, all_pop]
        dm_X_full = np.column_stack([demean_col(c, all_groups) for c in all_feat])
        dm_y_full = demean_col(all_fd, all_groups)
        pred_full = dm_X_full @ coeffs_final

        # ── Step 3: scatter plot ──────────────────────────────────────
        # Classify each observation for coloring
        reasons_arr = np.array(all_reasons)

        # Update clean observations that are statistical outliers
        clean_indices = np.where(reasons_arr == "clean")[0]
        stat_outlier_indices = clean_indices[is_stat_outlier]
        reasons_plot = reasons_arr.copy()
        # We don't overwrite — just note them. The "clean" ones that are
        # stat outliers stay blue but are mentioned in the count.

        # Plot each reason group
        counts = {}
        for reason in ["clean", "terrain_changed", "growth", "raided", "capped", "port"]:
            mask = reasons_arr == reason
            n_r = mask.sum()
            if n_r == 0:
                continue
            cfg = reason_config[reason]
            counts[reason] = n_r
            ax.scatter(pred_full[mask], dm_y_full[mask],
                      s=cfg["s"], alpha=cfg["alpha"], color=cfg["color"],
                      label=f'{cfg["label"]} ({n_r})',
                      zorder=cfg["zorder"], linewidths=0, rasterized=True)

        # Plot y=x line (perfect prediction)
        lim = max(abs(pred_full.min()), abs(pred_full.max()),
                  abs(dm_y_full.min()), abs(dm_y_full.max()))
        lim = min(lim, 0.4)
        ax.plot([-lim, lim], [-lim, lim], 'k-', lw=1.5, zorder=10, label='y=x')

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('Predicted (demeaned)')
        ax.set_ylabel('Actual (demeaned)')
        ax.set_title(f'{rid[:8]}  R²={r2_final:.3f} (clean, after outlier removal)')
        ax.set_aspect('equal')
        ax.legend(fontsize=6, loc='upper left', markerscale=2)

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'Food production: predicted vs actual (demeaned)\n'
        'food_delta = pl×(1−f) + fo×(1−f) + fb×f×(1−f) − pop + weather',
        fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig('food_proof.png', dpi=150, bbox_inches='tight')
    print("Saved food_proof.png")
    plt.close()


if __name__ == "__main__":
    main()
