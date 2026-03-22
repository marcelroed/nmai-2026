"""
Plot demeaned food_delta to visually identify the formula and outliers.

Reproduce:  python3 data-analysis/plot_food_curves.py
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
                grew = pop_delta < -0.05
                rows.append((
                    sb["population"], sb["food"],
                    *adj,
                    food_delta,
                    float(grew),
                    pop_delta,
                ))
                groups.append((seed, step))

    return np.array(rows, dtype=np.float64), groups


def demean_by_group(vals, groups):
    """Subtract group mean from each value."""
    group_ids = defaultdict(list)
    for i, g in enumerate(groups):
        group_ids[g].append(i)
    result = vals.copy()
    for indices in group_ids.values():
        result[indices] -= vals[indices].mean()
    return result


def main():
    rounds = load_replays_by_round()

    # Pick first 4 rounds for a 2x2 grid of plots
    round_ids = sorted(rounds.keys())[:4]

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 1: Demeaned food_delta vs food, colored by terrain score
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]
        arr, groups = collect_rows(replays)

        food = arr[:, 1]
        food_delta = arr[:, 6]
        n_pl, n_fo = arr[:, 2], arr[:, 3]
        grew = arr[:, 7] > 0.5

        food_delta_dm = demean_by_group(food_delta, groups)

        terrain_score = n_pl + 1.5 * n_fo

        # Plot non-growth as scatter, growth as different marker
        scatter = ax.scatter(food[~grew], food_delta_dm[~grew],
                           c=terrain_score[~grew], cmap='viridis',
                           s=1, alpha=0.15, vmin=0, vmax=12)
        ax.scatter(food[grew], food_delta_dm[grew],
                  c='red', s=3, alpha=0.3, marker='x', label='growth')

        ax.set_xlabel('food_before')
        ax.set_ylabel('food_delta (demeaned)')
        ax.set_title(f'Round {rid[:8]}')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_ylim(-0.5, 0.5)
        if idx == 0:
            ax.legend(markerscale=3)

    fig.colorbar(scatter, ax=axes, label='terrain_score (pl + 1.5×fo)', shrink=0.6)
    fig.suptitle('Demeaned food_delta vs food (color = terrain score, red x = growth events)',
                fontsize=14)
    plt.tight_layout()
    plt.savefig('food_delta_vs_food.png', dpi=150)
    print("Saved food_delta_vs_food.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 2: Demeaned food_delta vs food, BINNED by terrain score
    # Show median curves for different terrain levels
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]
        arr, groups = collect_rows(replays)

        food = arr[:, 1]
        food_delta = arr[:, 6]
        n_pl, n_fo = arr[:, 2], arr[:, 3]
        pop = arr[:, 0]
        grew = arr[:, 7] > 0.5

        food_delta_dm = demean_by_group(food_delta, groups)

        terrain_score = n_pl + 1.5 * n_fo

        # Exclude growth events
        mask = ~grew
        food_m = food[mask]
        fd_m = food_delta_dm[mask]
        ts_m = terrain_score[mask]
        pop_m = pop[mask]

        # Bin terrain score
        ts_bins = [(0, 3, 'C0'), (3, 5, 'C1'), (5, 7, 'C2'), (7, 9, 'C3'), (9, 15, 'C4')]
        food_bins = np.arange(0.025, 1.0, 0.05)

        for ts_lo, ts_hi, color in ts_bins:
            ts_mask = (ts_m >= ts_lo) & (ts_m < ts_hi)
            if ts_mask.sum() < 30:
                continue
            # Bin by food and compute median
            bin_centers = []
            bin_medians = []
            for fb in food_bins:
                fb_mask = ts_mask & (food_m >= fb - 0.025) & (food_m < fb + 0.025)
                if fb_mask.sum() >= 5:
                    bin_centers.append(fb)
                    bin_medians.append(np.median(fd_m[fb_mask]))
            if bin_centers:
                ax.plot(bin_centers, bin_medians, '-o', color=color, markersize=3,
                       label=f'terrain [{ts_lo},{ts_hi})')

        ax.set_xlabel('food_before')
        ax.set_ylabel('food_delta (demeaned, median)')
        ax.set_title(f'Round {rid[:8]} (no growth)')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.legend(fontsize=7)

    fig.suptitle('Median demeaned food_delta vs food, by terrain score (no growth events)',
                fontsize=14)
    plt.tight_layout()
    plt.savefig('food_curves_by_terrain.png', dpi=150)
    print("Saved food_curves_by_terrain.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 3: After subtracting terrain*(1-food) fit,
    # residual vs pop — looking for clean pop relationship
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]
        arr, groups = collect_rows(replays)

        food = arr[:, 1]
        pop = arr[:, 0]
        food_delta = arr[:, 6]
        n_pl, n_fo = arr[:, 2], arr[:, 3]
        grew = arr[:, 7] > 0.5
        omf = 1 - food

        food_delta_dm = demean_by_group(food_delta, groups)

        # Subtract terrain*(1-food) component using a simple fit
        # on non-growth data only
        mask = ~grew
        terrain_contrib = n_pl * omf  # will weight later
        terrain_contrib_fo = n_fo * omf

        # Quick fit of terrain coefficients on non-growth
        X_t = np.column_stack([
            (n_pl * omf)[mask].reshape(-1, 1),
            (n_fo * omf)[mask].reshape(-1, 1),
        ])
        X_t_dm = np.column_stack([
            demean_by_group((n_pl * omf)[mask], [groups[i] for i in range(len(groups)) if mask[i]]),
            demean_by_group((n_fo * omf)[mask], [groups[i] for i in range(len(groups)) if mask[i]]),
        ])
        y_t = food_delta_dm[mask]
        y_t_dm = demean_by_group(food_delta[mask], [groups[i] for i in range(len(groups)) if mask[i]])
        # Actually let me just use mean terrain coefficients
        # pl ≈ 0.06, fo ≈ 0.096
        terrain_effect = 0.06 * n_pl * omf + 0.096 * n_fo * omf
        terrain_effect_dm = demean_by_group(terrain_effect, groups)

        residual = food_delta_dm - terrain_effect_dm

        # Scatter: residual vs pop
        ax.scatter(pop[~grew], residual[~grew], s=1, alpha=0.1, c='C0', label='no growth')
        ax.scatter(pop[grew], residual[grew], s=3, alpha=0.3, c='red', marker='x', label='growth')
        ax.set_xlabel('pop_before')
        ax.set_ylabel('residual (after terrain)')
        ax.set_title(f'Round {rid[:8]}')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_ylim(-0.5, 0.5)
        if idx == 0:
            ax.legend(markerscale=3)

        # Add median line for non-growth
        pop_bins = np.arange(0.2, 3.5, 0.1)
        bin_centers = []
        bin_medians = []
        for pb in pop_bins:
            pb_mask = (~grew) & (pop >= pb - 0.05) & (pop < pb + 0.05)
            if pb_mask.sum() >= 10:
                bin_centers.append(pb)
                bin_medians.append(np.median(residual[pb_mask]))
        ax.plot(bin_centers, bin_medians, 'k-', linewidth=2, label='median (no growth)')

    fig.suptitle('Residual (after terrain×(1-food)) vs population', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_residual_vs_pop.png', dpi=150)
    print("Saved food_residual_vs_pop.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 4: After subtracting terrain*(1-food) AND pop effect,
    # residual vs food — looking for food*(1-food) pattern
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]
        arr, groups = collect_rows(replays)

        food = arr[:, 1]
        pop = arr[:, 0]
        food_delta = arr[:, 6]
        n_pl, n_fo = arr[:, 2], arr[:, 3]
        grew = arr[:, 7] > 0.5
        omf = 1 - food

        food_delta_dm = demean_by_group(food_delta, groups)

        # Subtract terrain*(1-food) + pop*coeff
        terrain_effect = 0.06 * n_pl * omf + 0.096 * n_fo * omf
        terrain_effect_dm = demean_by_group(terrain_effect, groups)
        pop_dm = demean_by_group(pop, groups)

        # Simple pop coefficient estimate (around -0.09)
        residual_after_terrain = food_delta_dm - terrain_effect_dm
        # Fit pop coefficient on non-growth
        mask = ~grew
        pop_coeff = np.sum(pop_dm[mask] * residual_after_terrain[mask]) / np.sum(pop_dm[mask]**2)

        residual = residual_after_terrain - pop_coeff * pop_dm

        # Scatter: residual vs food
        ax.scatter(food[~grew], residual[~grew], s=1, alpha=0.1, c='C0')
        ax.scatter(food[grew], residual[grew], s=3, alpha=0.3, c='red', marker='x')
        ax.set_xlabel('food_before')
        ax.set_ylabel(f'residual (after terrain+pop)')
        ax.set_title(f'Round {rid[:8]} (pop_coeff={pop_coeff:.4f})')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_ylim(-0.4, 0.4)

        # Add median lines for growth and non-growth
        food_bins = np.arange(0.05, 1.0, 0.05)
        for gmask, color, label in [(~grew, 'black', 'no growth'), (grew, 'red', 'growth')]:
            bin_centers = []
            bin_medians = []
            for fb in food_bins:
                fb_mask = gmask & (food >= fb - 0.025) & (food < fb + 0.025)
                if fb_mask.sum() >= 5:
                    bin_centers.append(fb)
                    bin_medians.append(np.median(residual[fb_mask]))
            if bin_centers:
                ax.plot(bin_centers, bin_medians, '-', color=color, linewidth=2, label=label)
        ax.legend(fontsize=7)

    fig.suptitle('Residual (after terrain×(1-food) + pop) vs food — looking for food feedback',
                fontsize=14)
    plt.tight_layout()
    plt.savefig('food_residual_vs_food.png', dpi=150)
    print("Saved food_residual_vs_food.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 5: Histogram of residuals (non-growth vs growth)
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays = rounds[rid]
        arr, groups = collect_rows(replays)

        food = arr[:, 1]
        pop = arr[:, 0]
        food_delta = arr[:, 6]
        n_pl, n_fo = arr[:, 2], arr[:, 3]
        grew = arr[:, 7] > 0.5
        omf = 1 - food

        food_delta_dm = demean_by_group(food_delta, groups)

        # Full model: terrain*(1-food) + pop
        terrain_effect = 0.06 * n_pl * omf + 0.096 * n_fo * omf
        terrain_effect_dm = demean_by_group(terrain_effect, groups)
        pop_dm = demean_by_group(pop, groups)

        mask = ~grew
        pop_coeff = np.sum(pop_dm[mask] * (food_delta_dm - terrain_effect_dm)[mask]) / np.sum(pop_dm[mask]**2)
        residual = food_delta_dm - terrain_effect_dm - pop_coeff * pop_dm

        ax.hist(residual[~grew], bins=100, range=(-0.4, 0.4), alpha=0.7,
               density=True, label=f'no growth (n={int((~grew).sum())})')
        ax.hist(residual[grew], bins=50, range=(-0.4, 0.4), alpha=0.5,
               color='red', density=True, label=f'growth (n={int(grew.sum())})')
        ax.set_xlabel('residual')
        ax.set_title(f'Round {rid[:8]}')
        ax.legend(fontsize=7)

    fig.suptitle('Distribution of residuals (after terrain×(1-food) + pop)', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_residual_histogram.png', dpi=150)
    print("Saved food_residual_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()
