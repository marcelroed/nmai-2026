"""
Clean food curve plots: control for terrain tightly, look for clean curves + outliers.

Reproduce:  python3 data-analysis/plot_food_clean.py
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
    """Count ALL terrain types in 8-connected neighbors."""
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

                adj = terrain_adj_full(grid, x, y)
                n_pl = adj.get(PLAINS, 0)
                n_fo = adj.get(FOREST, 0)
                n_mt = adj.get(MOUNTAIN, 0)
                n_oc = adj.get(OCEAN, 0)
                n_st = adj.get(SETTLEMENT, 0) + adj.get(PORT, 0) + adj.get(RUIN, 0)

                food_delta = sa["food"] - sb["food"]
                pop_delta = sa["population"] - sb["population"]
                grew = pop_delta < -0.05

                rows.append((
                    sb["population"], sb["food"],
                    n_pl, n_fo, n_mt, n_oc, n_st,
                    food_delta,
                    float(grew),
                    pop_delta,
                ))
                groups.append((seed, step))

    return np.array(rows, dtype=np.float64), groups


def demean_by_group(vals, groups):
    group_ids = defaultdict(list)
    for i, g in enumerate(groups):
        group_ids[g].append(i)
    result = vals.copy()
    for indices in group_ids.values():
        result[indices] -= vals[indices].mean()
    return result


def main():
    rounds = load_replays_by_round()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 1: For specific terrain combos, plot food_delta_dm vs food
    #           Split by growth/non-growth, show pop as color
    # ═══════════════════════════════════════════════════════════════════

    # Use first round with lots of data
    rid = sorted(rounds.keys())[0]
    replays = rounds[rid]
    arr, groups = collect_rows(replays)
    n = len(arr)

    pop = arr[:, 0]
    food = arr[:, 1]
    n_pl, n_fo, n_mt, n_oc, n_st = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
    food_delta = arr[:, 7]
    grew = arr[:, 8] > 0.5

    food_delta_dm = demean_by_group(food_delta, groups)

    # What terrain combos are most common?
    terrain_keys = {}
    for i in range(n):
        key = (int(n_pl[i]), int(n_fo[i]), int(n_mt[i]), int(n_oc[i]), int(n_st[i]))
        if key not in terrain_keys:
            terrain_keys[key] = 0
        terrain_keys[key] += 1

    print(f"Round {rid[:12]}: {n} observations")
    print(f"Top terrain combos (pl, fo, mt, oc, st):")
    top_combos = sorted(terrain_keys.items(), key=lambda x: -x[1])[:20]
    for combo, count in top_combos:
        print(f"  {combo}: {count} obs")

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 1: For a few common terrain combos, plot food_delta_dm vs food
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    axes = axes.flatten()

    combos_to_plot = [c for c, cnt in top_combos[:16]]

    for idx, combo in enumerate(combos_to_plot):
        ax = axes[idx]
        mask = np.array([
            (int(n_pl[i]), int(n_fo[i]), int(n_mt[i]), int(n_oc[i]), int(n_st[i])) == combo
            for i in range(n)
        ])

        ng = mask & (~grew)
        g = mask & grew

        if ng.sum() > 0:
            sc = ax.scatter(food[ng], food_delta_dm[ng], c=pop[ng], cmap='viridis',
                          s=5, alpha=0.4, vmin=0, vmax=3)
        if g.sum() > 0:
            ax.scatter(food[g], food_delta_dm[g], c='red', s=10, alpha=0.5, marker='x')

        ax.set_title(f'pl={combo[0]} fo={combo[1]} mt={combo[2]} oc={combo[3]} st={combo[4]}\n'
                    f'n={mask.sum()} (grew={g.sum()})', fontsize=9)
        ax.set_xlabel('food')
        ax.set_ylabel('food_delta (dm)')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_ylim(-0.4, 0.4)

    fig.suptitle(f'Round {rid[:12]}: food_delta_dm vs food by terrain combo\n'
                f'(color=pop, red x=growth)', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_by_terrain_combo.png', dpi=150)
    print("Saved food_by_terrain_combo.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 2: For non-growth observations, group by (pl, fo) only,
    #           narrow pop bin, plot food_delta_dm vs food
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    pop_ranges = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.4), (1.4, 2.0), (2.0, 5.0)]

    for pidx, (plo, phi) in enumerate(pop_ranges):
        ax = axes[pidx]
        pop_mask = (~grew) & (pop >= plo) & (pop < phi)

        # Group by (pl, fo) = terrain that matters
        for npl in range(8):
            for nfo in range(8):
                tmask = pop_mask & (n_pl == npl) & (n_fo == nfo)
                if tmask.sum() < 20:
                    continue
                # Bin by food and get median
                food_bins = np.arange(0.05, 1.0, 0.05)
                centers = []
                medians = []
                for fb in food_bins:
                    fmask = tmask & (food >= fb - 0.025) & (food < fb + 0.025)
                    if fmask.sum() >= 3:
                        centers.append(fb)
                        medians.append(np.median(food_delta_dm[fmask]))
                if len(centers) >= 3:
                    ax.plot(centers, medians, '-o', markersize=2,
                           label=f'pl={npl} fo={nfo}', alpha=0.7)

        ax.set_title(f'pop ∈ [{plo}, {phi})')
        ax.set_xlabel('food')
        ax.set_ylabel('food_delta (dm, median)')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(-0.3, 0.3)

    fig.suptitle(f'Round {rid[:12]}: Median food_delta_dm vs food\n'
                f'by terrain (pl,fo) and pop bin (non-growth only)', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_curves_controlled.png', dpi=150)
    print("Saved food_curves_controlled.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 3: ALL rounds pooled — for fixed (pl,fo), non-growth,
    #           mid-pop (0.7-1.3), scatter + quantiles of food_delta_dm vs food
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    axes = axes.flatten()

    # Common (pl, fo) combos across all rounds
    pf_combos = [(1, 4), (2, 3), (2, 4), (3, 2), (3, 3), (0, 5),
                 (1, 5), (4, 2), (2, 5), (3, 4), (0, 6), (1, 3)]

    for cidx, (target_pl, target_fo) in enumerate(pf_combos):
        ax = axes[cidx]

        all_food = []
        all_fd_dm = []

        for rid2 in sorted(rounds):
            replays2 = rounds[rid2]
            arr2, groups2 = collect_rows(replays2)
            pop2 = arr2[:, 0]
            food2 = arr2[:, 1]
            npl2, nfo2 = arr2[:, 2], arr2[:, 3]
            fd2 = arr2[:, 7]
            grew2 = arr2[:, 8] > 0.5

            fd_dm2 = demean_by_group(fd2, groups2)

            mask2 = (~grew2) & (npl2 == target_pl) & (nfo2 == target_fo) & \
                    (pop2 >= 0.7) & (pop2 < 1.3)

            all_food.extend(food2[mask2])
            all_fd_dm.extend(fd_dm2[mask2])

        all_food = np.array(all_food)
        all_fd_dm = np.array(all_fd_dm)

        if len(all_food) < 20:
            ax.set_title(f'pl={target_pl} fo={target_fo} (too few)')
            continue

        ax.scatter(all_food, all_fd_dm, s=1, alpha=0.15)

        # Quantile curves
        food_bins = np.arange(0.05, 1.0, 0.05)
        centers, q25, q50, q75 = [], [], [], []
        for fb in food_bins:
            fmask = (all_food >= fb - 0.03) & (all_food < fb + 0.03)
            if fmask.sum() >= 5:
                centers.append(fb)
                q25.append(np.percentile(all_fd_dm[fmask], 25))
                q50.append(np.median(all_fd_dm[fmask]))
                q75.append(np.percentile(all_fd_dm[fmask], 75))

        if centers:
            ax.fill_between(centers, q25, q75, alpha=0.3, color='C1')
            ax.plot(centers, q50, 'C1-', linewidth=2)

        ax.set_title(f'pl={target_pl} fo={target_fo} (n={len(all_food)})')
        ax.set_xlabel('food')
        ax.set_ylabel('food_delta (dm)')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_ylim(-0.3, 0.3)

    fig.suptitle('All rounds pooled: food_delta_dm vs food\n'
                '(non-growth, pop ∈ [0.7, 1.3], specific terrain)\n'
                'Orange = median + IQR', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_curves_pooled.png', dpi=150)
    print("Saved food_curves_pooled.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 4: Pop effect — for fixed terrain combo, non-growth,
    #           plot food_delta_dm vs pop, colored by food
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid3 in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays3 = rounds[rid3]
        arr3, groups3 = collect_rows(replays3)

        pop3 = arr3[:, 0]
        food3 = arr3[:, 1]
        npl3, nfo3 = arr3[:, 2], arr3[:, 3]
        fd3 = arr3[:, 7]
        grew3 = arr3[:, 8] > 0.5
        omf3 = 1 - food3

        fd_dm3 = demean_by_group(fd3, groups3)

        # Subtract terrain effect (using approx coefficients)
        terrain_eff = 0.06 * npl3 * omf3 + 0.096 * nfo3 * omf3
        terrain_dm = demean_by_group(terrain_eff, groups3)
        residual = fd_dm3 - terrain_dm

        # Non-growth only
        ng3 = ~grew3

        ax.scatter(pop3[ng3], residual[ng3], c=food3[ng3], cmap='coolwarm',
                  s=1, alpha=0.15, vmin=0, vmax=1)
        ax.scatter(pop3[grew3], residual[grew3], c='black', s=5, alpha=0.3, marker='x')

        # Median in pop bins, split by food level
        for flo, fhi, color, label in [(0.3, 0.6, 'blue', 'f=[.3,.6)'),
                                        (0.6, 0.8, 'green', 'f=[.6,.8)'),
                                        (0.8, 1.0, 'red', 'f=[.8,1)')]:
            pbins = np.arange(0.3, 3.5, 0.15)
            centers, medians = [], []
            for pb in pbins:
                pmask = ng3 & (pop3 >= pb - 0.075) & (pop3 < pb + 0.075) & \
                        (food3 >= flo) & (food3 < fhi)
                if pmask.sum() >= 5:
                    centers.append(pb)
                    medians.append(np.median(residual[pmask]))
            if centers:
                ax.plot(centers, medians, '-', color=color, linewidth=2, label=label)

        ax.set_xlabel('pop_before')
        ax.set_ylabel('residual (after terrain)')
        ax.set_title(f'Round {rid3[:8]}')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_ylim(-0.4, 0.4)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Residual (after terrain×(1-food)) vs population\n'
                '(non-growth scatter, black x=growth, lines=median by food level)',
                fontsize=14)
    plt.tight_layout()
    plt.savefig('food_pop_by_food_level.png', dpi=150)
    print("Saved food_pop_by_food_level.png")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE 5: Do adjacent settlements (n_st) matter?
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()

    for idx, rid4 in enumerate(sorted(rounds)):
        ax = axes[idx]
        replays4 = rounds[rid4]
        arr4, groups4 = collect_rows(replays4)

        pop4 = arr4[:, 0]
        food4 = arr4[:, 1]
        npl4, nfo4 = arr4[:, 2], arr4[:, 3]
        nst4 = arr4[:, 6]
        fd4 = arr4[:, 7]
        grew4 = arr4[:, 8] > 0.5
        omf4 = 1 - food4

        fd_dm4 = demean_by_group(fd4, groups4)

        # Subtract terrain + pop effect
        terrain_eff = 0.06 * npl4 * omf4 + 0.096 * nfo4 * omf4
        terrain_dm = demean_by_group(terrain_eff, groups4)
        pop_dm4 = demean_by_group(pop4, groups4)
        residual4 = fd_dm4 - terrain_dm + 0.09 * pop_dm4  # add back pop (coeff ~ -0.09)

        ng4 = ~grew4

        # Box plot by n_st
        st_vals = sorted(set(int(x) for x in nst4))
        data_by_st = []
        labels = []
        for sv in st_vals:
            mask = ng4 & (nst4 == sv)
            if mask.sum() >= 20:
                data_by_st.append(residual4[mask])
                labels.append(f'{sv}\n(n={mask.sum()})')

        if data_by_st:
            bp = ax.boxplot(data_by_st, labels=labels, showfliers=False)
            ax.set_xlabel('n_adjacent_settlements')
            ax.set_ylabel('residual')
            ax.set_title(f'Round {rid4[:8]}')
            ax.axhline(y=0, color='gray', linewidth=0.5)

    fig.suptitle('Effect of adjacent settlements on food residual\n'
                '(after terrain×(1-food) + pop, non-growth)', fontsize=14)
    plt.tight_layout()
    plt.savefig('food_settlement_effect.png', dpi=150)
    print("Saved food_settlement_effect.png")
    plt.close()


if __name__ == "__main__":
    main()
