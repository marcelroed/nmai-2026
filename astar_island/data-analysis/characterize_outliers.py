"""
Characterize the ~20% outliers that don't fit the food production formula.

After filtering raided/capped/growth observations and fitting:
  food_delta = terrain*(1-food) + feedback*food*(1-food) - pop_rate*pop + weather
R² is 0.72-0.89. After removing ~20% outliers, R² → 0.95-0.98.

What makes those 20% different?

Hypotheses:
1. Terrain changed between frames (neighbor died/grew → grid changed)
2. Settlement is adjacent to a port (trade effect)
3. Settlement is adjacent to a settlement that was raided
4. Some step-specific effect (winter?)
5. Population-level threshold effect

Reproduce:  python3 data-analysis/characterize_outliers.py
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


def collect_detailed_rows(replays):
    """Collect rows with extra detail for outlier characterization."""
    rows = []
    meta = []  # extra info per row for diagnosis
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

            # Count deaths and births this step
            after_all = {(s["x"], s["y"]): s for s in fa["settlements"]}
            before_all = {(s["x"], s["y"]): s for s in fb["settlements"]}

            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                if grid_before[y][x] == PORT:
                    continue

                def_delta = sa["defense"] - sb["defense"]
                pop_delta = sa["population"] - sb["population"]
                food_after = sa["food"]
                raided = def_delta < -0.001
                capped_high = food_after >= 0.997
                capped_low = food_after <= 0.001
                grew = pop_delta < -0.05

                if raided or capped_high or capped_low or grew:
                    continue

                adj_before = terrain_adj_full(grid_before, x, y)
                adj_after = terrain_adj_full(grid_after, x, y)

                # Did terrain change?
                terrain_changed = adj_before != adj_after

                # Count specific terrain changes
                n_sett_before = adj_before.get(SETTLEMENT, 0) + adj_before.get(PORT, 0)
                n_sett_after = adj_after.get(SETTLEMENT, 0) + adj_after.get(PORT, 0)
                n_ruin_before = adj_before.get(RUIN, 0)
                n_ruin_after = adj_after.get(RUIN, 0)

                # Is this settlement adjacent to a port?
                adj_port = adj_before.get(PORT, 0) > 0

                # Was any neighbor raided?
                neighbor_raided = False
                for ny in range(max(0, y - 1), min(len(grid_before), y + 2)):
                    for nx in range(max(0, x - 1), min(len(grid_before[0]), x + 2)):
                        if nx == x and ny == y:
                            continue
                        npos = (nx, ny)
                        nb = {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}.get(npos)
                        na = after_map.get(npos)
                        if nb and na and na["owner_id"] == nb["owner_id"]:
                            if na["defense"] - nb["defense"] < -0.001:
                                neighbor_raided = True

                food_delta = sa["food"] - sb["food"]
                omf = 1 - sb["food"]

                rows.append((
                    sb["population"], sb["food"],
                    adj_before.get(PLAINS, 0), adj_before.get(FOREST, 0),
                    adj_before.get(MOUNTAIN, 0), adj_before.get(OCEAN, 0),
                    adj_before.get(SETTLEMENT, 0) + adj_before.get(PORT, 0) + adj_before.get(RUIN, 0),
                    food_delta,
                ))
                meta.append({
                    "seed": seed, "step": step,
                    "x": x, "y": y,
                    "terrain_changed": terrain_changed,
                    "n_sett_delta": n_sett_after - n_sett_before,
                    "n_ruin_delta": n_ruin_after - n_ruin_before,
                    "adj_port": adj_port,
                    "neighbor_raided": neighbor_raided,
                    "pop_delta": pop_delta,
                    "def_delta": def_delta,
                    "food_after": food_after,
                    "wealth": sb["wealth"],
                    "defense": sb["defense"],
                })

    return np.array(rows, dtype=np.float64), meta


def main():
    rounds = load_replays_by_round()

    # Use first round for detailed analysis, then check if patterns hold
    all_outlier_fracs = defaultdict(list)

    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()

    fig2, axes2 = plt.subplots(3, 3, figsize=(22, 18))
    axes2 = axes2.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        replays = rounds[rid]
        arr, meta = collect_detailed_rows(replays)
        n = len(arr)

        pop = arr[:, 0]
        food = arr[:, 1]
        n_pl, n_fo, n_mt, n_oc, n_st = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
        food_delta = arr[:, 7]
        omf = 1 - food

        groups = [(m["seed"], m["step"]) for m in meta]

        # Fit the model
        features = np.column_stack([
            n_pl * omf, n_fo * omf,
            food * omf,
            pop,
            food_delta,
        ])
        dm = np.zeros_like(features)
        for col in range(features.shape[1]):
            dm[:, col] = demean_col(features[:, col], groups)

        c, r2, mae, res = fit_ols(dm[:, :4], dm[:, 4])

        # Classify outliers: |residual| > 3*MAE
        threshold = 3 * mae
        is_outlier = np.abs(res) > threshold

        n_out = is_outlier.sum()
        print(f"\n{'='*72}")
        print(f"Round {rid[:12]}: {n} clean obs, {n_out} outliers ({100*n_out/n:.1f}%)")
        print(f"  Model R²={r2:.4f}, MAE={mae:.6f}")
        print(f"  Coeffs: pl*(1-f)={c[0]:.5f} fo*(1-f)={c[1]:.5f} f*(1-f)={c[2]:.5f} pop={c[3]:.5f}")

        # Extract meta arrays
        terrain_changed = np.array([m["terrain_changed"] for m in meta])
        adj_port = np.array([m["adj_port"] for m in meta])
        neighbor_raided = np.array([m["neighbor_raided"] for m in meta])
        n_sett_delta = np.array([m["n_sett_delta"] for m in meta])
        steps = np.array([m["step"] for m in meta])
        wealth = np.array([m["wealth"] for m in meta])
        defense = np.array([m["defense"] for m in meta])

        # ── Characterize outliers vs inliers ────────────────────────────
        def compare(label, mask):
            frac_out = mask[is_outlier].mean() if is_outlier.sum() > 0 else 0
            frac_in = mask[~is_outlier].mean() if (~is_outlier).sum() > 0 else 0
            ratio = frac_out / frac_in if frac_in > 0 else float('inf')
            print(f"  {label}: outliers={100*frac_out:.1f}% inliers={100*frac_in:.1f}% (ratio={ratio:.2f})")
            all_outlier_fracs[label].append((frac_out, frac_in, ratio))

        compare("terrain_changed", terrain_changed)
        compare("adj_port", adj_port)
        compare("neighbor_raided", neighbor_raided)
        compare("sett_appeared", n_sett_delta > 0)
        compare("sett_disappeared", n_sett_delta < 0)

        # Step distribution
        print(f"  Step distribution:")
        print(f"    Outlier mean step: {steps[is_outlier].mean():.1f} vs inlier: {steps[~is_outlier].mean():.1f}")

        # Food level distribution
        print(f"  Food level:")
        print(f"    Outlier mean food: {food[is_outlier].mean():.3f} vs inlier: {food[~is_outlier].mean():.3f}")

        # Pop distribution
        print(f"  Population:")
        print(f"    Outlier mean pop: {pop[is_outlier].mean():.3f} vs inlier: {pop[~is_outlier].mean():.3f}")

        # Wealth distribution
        print(f"  Wealth:")
        print(f"    Outlier mean wealth: {wealth[is_outlier].mean():.3f} vs inlier: {wealth[~is_outlier].mean():.3f}")

        # Residual sign
        pos_res = res[is_outlier] > 0
        print(f"  Residual sign: {pos_res.sum()} positive, {(~pos_res).sum()} negative")

        # ── Plot 1: Residual histogram colored by terrain_changed ────────
        ax = axes[idx]
        inlier_tc = res[~is_outlier & terrain_changed]
        inlier_no = res[~is_outlier & ~terrain_changed]
        outlier_tc = res[is_outlier & terrain_changed]
        outlier_no = res[is_outlier & ~terrain_changed]

        ax.hist(res[~terrain_changed], bins=200, range=(-0.15, 0.15),
                density=True, alpha=0.6, label=f'terrain stable ({(~terrain_changed).sum()})', color='blue')
        ax.hist(res[terrain_changed], bins=200, range=(-0.15, 0.15),
                density=True, alpha=0.6, label=f'terrain changed ({terrain_changed.sum()})', color='red')
        ax.axvline(x=threshold, color='gray', ls='--', lw=0.5)
        ax.axvline(x=-threshold, color='gray', ls='--', lw=0.5)
        ax.set_title(f'Round {rid[:8]}')
        ax.set_xlabel('residual')
        ax.legend(fontsize=7)

        # ── Plot 2: Outlier rate by step ──────────────────────────────────
        ax2 = axes2[idx]
        step_vals = sorted(set(steps))
        out_rates = []
        tc_rates = []
        for sv in step_vals:
            mask = steps == sv
            if mask.sum() > 0:
                out_rates.append(is_outlier[mask].mean())
                tc_rates.append(terrain_changed[mask].mean())
            else:
                out_rates.append(0)
                tc_rates.append(0)
        ax2.plot(step_vals, out_rates, 'b.-', label='outlier rate', alpha=0.7)
        ax2.plot(step_vals, tc_rates, 'r.-', label='terrain changed rate', alpha=0.7)
        ax2.set_xlabel('step')
        ax2.set_ylabel('rate')
        ax2.set_title(f'Round {rid[:8]}')
        ax2.legend(fontsize=7)

    fig.suptitle('Residual distribution: terrain stable vs changed', fontsize=14)
    plt.figure(fig)
    plt.tight_layout()
    fig.savefig('outlier_terrain_changed.png', dpi=150)
    print("\nSaved outlier_terrain_changed.png")

    fig2.suptitle('Outlier rate and terrain change rate by step', fontsize=14)
    plt.figure(fig2)
    plt.tight_layout()
    fig2.savefig('outlier_rate_by_step.png', dpi=150)
    print("Saved outlier_rate_by_step.png")
    plt.close('all')

    # ═══════════════════════════════════════════════════════════════════════
    # Summary across rounds
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("CROSS-ROUND SUMMARY")
    print(f"{'='*72}")
    for label, vals in all_outlier_fracs.items():
        ratios = [v[2] for v in vals]
        print(f"  {label}: mean ratio = {np.mean(ratios):.2f} "
              f"(outlier frac={np.mean([v[0] for v in vals])*100:.1f}% "
              f"vs inlier frac={np.mean([v[1] for v in vals])*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2: Re-fit excluding terrain-changed observations
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("PART 2: Re-fit excluding terrain-changed observations")
    print(f"{'='*72}")

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, meta = collect_detailed_rows(replays)
        n = len(arr)

        terrain_changed = np.array([m["terrain_changed"] for m in meta])
        stable = ~terrain_changed

        pop = arr[stable, 0]
        food = arr[stable, 1]
        n_pl = arr[stable, 2]
        n_fo = arr[stable, 3]
        food_delta = arr[stable, 7]
        omf = 1 - food

        groups = [(meta[i]["seed"], meta[i]["step"]) for i in range(n) if stable[i]]

        features = np.column_stack([
            n_pl * omf, n_fo * omf,
            food * omf,
            pop,
            food_delta,
        ])
        dm = np.zeros_like(features)
        for col in range(features.shape[1]):
            dm[:, col] = demean_col(features[:, col], groups)

        c, r2, mae, res = fit_ols(dm[:, :4], dm[:, 4])

        # Now check outlier rate on this subset
        threshold = 3 * mae
        is_outlier = np.abs(res) > threshold
        n_out = is_outlier.sum()

        print(f"\n  Round {rid[:12]}: {stable.sum()} stable obs (removed {terrain_changed.sum()})")
        print(f"    R²={r2:.6f}, MAE={mae:.6f}, outliers={n_out} ({100*n_out/stable.sum():.1f}%)")
        print(f"    pl*(1-f)={c[0]:.6f} fo*(1-f)={c[1]:.6f} f*(1-f)={c[2]:.6f} pop={c[3]:.6f}")

        # After one round of outlier removal
        inlier = ~is_outlier
        pop2 = pop[inlier]
        food2 = food[inlier]
        npl2 = n_pl[inlier]
        nfo2 = n_fo[inlier]
        fd2 = food_delta[inlier]
        omf2 = 1 - food2
        groups2 = [groups[i] for i in range(len(groups)) if inlier[i]]

        features2 = np.column_stack([
            npl2 * omf2, nfo2 * omf2,
            food2 * omf2,
            pop2,
            fd2,
        ])
        dm2 = np.zeros_like(features2)
        for col in range(features2.shape[1]):
            dm2[:, col] = demean_col(features2[:, col], groups2)

        c2, r2_2, mae2, _ = fit_ols(dm2[:, :4], dm2[:, 4])
        print(f"    After 1 outlier pass: R²={r2_2:.6f}, MAE={mae2:.6f}")
        print(f"    pl*(1-f)={c2[0]:.6f} fo*(1-f)={c2[1]:.6f} f*(1-f)={c2[2]:.6f} pop={c2[3]:.6f}")


if __name__ == "__main__":
    main()
