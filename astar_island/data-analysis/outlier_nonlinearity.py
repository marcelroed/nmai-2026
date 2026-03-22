"""
Investigate outlier nonlinearity: outliers have low food + high pop + negative residuals.

Hypotheses:
A) Pop consumption is nonlinear: pop_cost = a*pop + b*pop² or a*pop*food
B) There's a starvation threshold: extra consumption when food < threshold
C) The food*(1-food) feedback term has the wrong functional form
D) Growth events are being missed (threshold too lenient)

Strategy: On terrain-stable data, plot residual vs food for different pop bins,
and residual vs pop for different food bins.

Reproduce:  python3 data-analysis/outlier_nonlinearity.py
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
    """Collect rows where terrain didn't change between frames."""
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

                # Skip raided, capped, grew
                if sa["defense"] - sb["defense"] < -0.001:
                    continue
                if sa["food"] >= 0.997 or sa["food"] <= 0.001:
                    continue
                if sa["population"] - sb["population"] < -0.05:
                    continue

                # Skip if terrain changed
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
                    sa["population"] - sb["population"],
                    sb["defense"], sb["wealth"],
                ))
                groups.append((seed, step))

    return np.array(rows, dtype=np.float64), groups


def main():
    rounds = load_replays_by_round()

    # Pool all rounds for maximum signal
    all_rows = []
    all_groups = []
    round_ids = []

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        # Tag groups with round_id to keep them separate
        tagged = [(rid, s, st) for s, st in groups]
        all_rows.append(arr)
        all_groups.extend(tagged)
        round_ids.extend([rid] * len(arr))

    arr = np.vstack(all_rows)
    groups = all_groups
    n = len(arr)
    print(f"Total terrain-stable clean observations: {n}")

    pop = arr[:, 0]
    food = arr[:, 1]
    n_pl, n_fo, n_mt, n_oc, n_st = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
    food_delta = arr[:, 7]
    pop_delta = arr[:, 8]
    defense = arr[:, 9]
    wealth = arr[:, 10]
    omf = 1 - food

    # ═══════════════════════════════════════════════════════════════════
    # Fit base model
    # ═══════════════════════════════════════════════════════════════════
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
    print(f"\nBase model R²={r2:.6f}, MAE={mae:.6f}")
    print(f"  pl*(1-f)={c[0]:.6f} fo*(1-f)={c[1]:.6f} f*(1-f)={c[2]:.6f} pop={c[3]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # Test extended models
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Extended models ---")

    # Model A: Add pop²
    features_a = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop, pop**2, food_delta,
    ])
    dm_a = np.zeros_like(features_a)
    for col in range(features_a.shape[1]):
        dm_a[:, col] = demean_col(features_a[:, col], groups)
    c_a, r2_a, mae_a, res_a = fit_ols(dm_a[:, :5], dm_a[:, 5])
    print(f"A) +pop²: R²={r2_a:.6f} MAE={mae_a:.6f}  pop={c_a[3]:.6f} pop²={c_a[4]:.6f}")

    # Model B: Add pop*food
    features_b = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop, pop * food, food_delta,
    ])
    dm_b = np.zeros_like(features_b)
    for col in range(features_b.shape[1]):
        dm_b[:, col] = demean_col(features_b[:, col], groups)
    c_b, r2_b, mae_b, res_b = fit_ols(dm_b[:, :5], dm_b[:, 5])
    print(f"B) +pop*food: R²={r2_b:.6f} MAE={mae_b:.6f}  pop={c_b[3]:.6f} pop*food={c_b[4]:.6f}")

    # Model C: Add pop*(1-food)
    features_c = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop * omf, food_delta,
    ])
    dm_c = np.zeros_like(features_c)
    for col in range(features_c.shape[1]):
        dm_c[:, col] = demean_col(features_c[:, col], groups)
    c_c, r2_c, mae_c, res_c = fit_ols(dm_c[:, :4], dm_c[:, 4])
    print(f"C) pop*(1-f): R²={r2_c:.6f} MAE={mae_c:.6f}  pop*(1-f)={c_c[3]:.6f}")

    # Model D: Replace pop with pop*food
    features_d = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop * food, food_delta,
    ])
    dm_d = np.zeros_like(features_d)
    for col in range(features_d.shape[1]):
        dm_d[:, col] = demean_col(features_d[:, col], groups)
    c_d, r2_d, mae_d, res_d = fit_ols(dm_d[:, :4], dm_d[:, 4])
    print(f"D) pop*food instead of pop: R²={r2_d:.6f} MAE={mae_d:.6f}  pop*f={c_d[3]:.6f}")

    # Model E: Both pop and pop*food
    features_e = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop, pop * food, food_delta,
    ])
    dm_e = np.zeros_like(features_e)
    for col in range(features_e.shape[1]):
        dm_e[:, col] = demean_col(features_e[:, col], groups)
    c_e, r2_e, mae_e, res_e = fit_ols(dm_e[:, :5], dm_e[:, 5])
    print(f"E) pop + pop*food: R²={r2_e:.6f} MAE={mae_e:.6f}  pop={c_e[3]:.6f} pop*f={c_e[4]:.6f}")

    # Model F: Add food² (quadratic food effect)
    features_f = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop, food**2, food_delta,
    ])
    dm_f = np.zeros_like(features_f)
    for col in range(features_f.shape[1]):
        dm_f[:, col] = demean_col(features_f[:, col], groups)
    c_f, r2_f, mae_f, res_f = fit_ols(dm_f[:, :5], dm_f[:, 5])
    print(f"F) +food²: R²={r2_f:.6f} MAE={mae_f:.6f}  f²={c_f[4]:.6f}")

    # Model G: terrain*(1-food) + pop + pop² + pop*food
    features_g = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop, pop**2, pop * food, food_delta,
    ])
    dm_g = np.zeros_like(features_g)
    for col in range(features_g.shape[1]):
        dm_g[:, col] = demean_col(features_g[:, col], groups)
    c_g, r2_g, mae_g, res_g = fit_ols(dm_g[:, :6], dm_g[:, 6])
    print(f"G) pop + pop² + pop*food: R²={r2_g:.6f} MAE={mae_g:.6f}")
    print(f"   pop={c_g[3]:.6f} pop²={c_g[4]:.6f} pop*f={c_g[5]:.6f}")

    # Model H: Everything logistic: terrain*(1-f) + f*(1-f) + pop*(1-f)  [all scaled by (1-f)]
    features_h = np.column_stack([
        n_pl * omf, n_fo * omf, food * omf, pop * omf, food_delta,
    ])
    dm_h = np.zeros_like(features_h)
    for col in range(features_h.shape[1]):
        dm_h[:, col] = demean_col(features_h[:, col], groups)
    c_h, r2_h, mae_h, _ = fit_ols(dm_h[:, :4], dm_h[:, 4])
    print(f"H) Everything × (1-f): R²={r2_h:.6f} MAE={mae_h:.6f}")
    print(f"   pl*(1-f)={c_h[0]:.6f} fo*(1-f)={c_h[1]:.6f} f*(1-f)={c_h[2]:.6f} pop*(1-f)={c_h[3]:.6f}")

    # Model I: Try (terrain + food_feedback*food - pop_rate*pop) * (1-food)
    # i.e., food_delta = (a*pl + b*fo + c*f - d*pop) * (1-f) + weather
    # This means: terrain*(1-f) + food*(1-f) - pop*(1-f)
    # All three terms scale with (1-f). Let's see if pop*(1-f) is better.
    features_i = np.column_stack([
        (n_pl + 1.5 * n_fo) * omf,  # single terrain score
        food * omf,
        pop * omf,
        pop,
        food_delta,
    ])
    dm_i = np.zeros_like(features_i)
    for col in range(features_i.shape[1]):
        dm_i[:, col] = demean_col(features_i[:, col], groups)
    c_i, r2_i, mae_i, _ = fit_ols(dm_i[:, :4], dm_i[:, 4])
    print(f"I) terrain*(1-f) + f*(1-f) + pop*(1-f) + pop: R²={r2_i:.6f} MAE={mae_i:.6f}")
    print(f"   terrain*(1-f)={c_i[0]:.6f} f*(1-f)={c_i[1]:.6f} pop*(1-f)={c_i[2]:.6f} pop={c_i[3]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PLOTS: Residual patterns
    # ═══════════════════════════════════════════════════════════════════

    # Use the best model for residual analysis
    print("\n\nUsing base model for residual plots...")
    best_res = res

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Residual vs food, colored by pop bin
    ax = axes[0, 0]
    pop_bins = [(0, 0.6, 'blue'), (0.6, 1.0, 'green'), (1.0, 1.5, 'orange'), (1.5, 5.0, 'red')]
    for lo, hi, col in pop_bins:
        mask = (pop >= lo) & (pop < hi)
        if mask.sum() > 0:
            # Bin by food and plot median residual
            food_vals = food[mask]
            res_vals = best_res[mask]
            bins = np.linspace(0, 1, 51)
            centers = (bins[:-1] + bins[1:]) / 2
            medians = []
            for j in range(len(bins) - 1):
                bm = (food_vals >= bins[j]) & (food_vals < bins[j+1])
                if bm.sum() >= 5:
                    medians.append(np.median(res_vals[bm]))
                else:
                    medians.append(np.nan)
            ax.plot(centers, medians, '.-', color=col, label=f'pop=[{lo},{hi})', alpha=0.8)
    ax.axhline(y=0, color='gray', lw=0.5)
    ax.set_xlabel('food')
    ax.set_ylabel('median residual')
    ax.set_title('Median residual vs food (by pop bin)')
    ax.legend(fontsize=7)

    # 2. Residual vs pop, colored by food bin
    ax = axes[0, 1]
    food_bins = [(0, 0.3, 'purple'), (0.3, 0.6, 'blue'), (0.6, 0.8, 'green'), (0.8, 1.0, 'orange')]
    for lo, hi, col in food_bins:
        mask = (food >= lo) & (food < hi)
        if mask.sum() > 0:
            pop_vals = pop[mask]
            res_vals = best_res[mask]
            bins = np.linspace(0, 3, 31)
            centers = (bins[:-1] + bins[1:]) / 2
            medians = []
            for j in range(len(bins) - 1):
                bm = (pop_vals >= bins[j]) & (pop_vals < bins[j+1])
                if bm.sum() >= 5:
                    medians.append(np.median(res_vals[bm]))
                else:
                    medians.append(np.nan)
            ax.plot(centers, medians, '.-', color=col, label=f'food=[{lo},{hi})', alpha=0.8)
    ax.axhline(y=0, color='gray', lw=0.5)
    ax.set_xlabel('population')
    ax.set_ylabel('median residual')
    ax.set_title('Median residual vs pop (by food bin)')
    ax.legend(fontsize=7)

    # 3. Scatter: food vs pop, colored by residual sign/magnitude
    ax = axes[0, 2]
    # Only plot a subsample for visibility
    np.random.seed(42)
    idx = np.random.choice(n, min(5000, n), replace=False)
    sc = ax.scatter(food[idx], pop[idx], c=best_res[idx], s=2, alpha=0.3,
                   cmap='RdBu', vmin=-0.1, vmax=0.1)
    plt.colorbar(sc, ax=ax, label='residual')
    ax.set_xlabel('food')
    ax.set_ylabel('population')
    ax.set_title('Residual in food-pop space')

    # 4. Residual histogram overall
    ax = axes[1, 0]
    ax.hist(best_res, bins=300, range=(-0.2, 0.2), density=True, alpha=0.7)
    ax.axvline(x=0, color='gray', lw=0.5)
    ax.set_xlabel('residual')
    ax.set_title('Overall residual distribution')

    # 5. Residual vs pop_delta (checking for sub-threshold growth)
    ax = axes[1, 1]
    # Bin pop_delta and show median residual
    pd_bins = np.linspace(-0.05, 0.5, 56)
    pd_centers = (pd_bins[:-1] + pd_bins[1:]) / 2
    pd_medians = []
    pd_counts = []
    for j in range(len(pd_bins) - 1):
        bm = (pop_delta >= pd_bins[j]) & (pop_delta < pd_bins[j+1])
        pd_counts.append(bm.sum())
        if bm.sum() >= 5:
            pd_medians.append(np.median(best_res[bm]))
        else:
            pd_medians.append(np.nan)
    ax.plot(pd_centers, pd_medians, 'b.-', alpha=0.7)
    ax.axhline(y=0, color='gray', lw=0.5)
    ax.set_xlabel('pop_delta')
    ax.set_ylabel('median residual')
    ax.set_title('Median residual vs pop_delta')

    # 6. Residual vs n_st (number of adjacent settlements)
    ax = axes[1, 2]
    for sv in sorted(set(n_st.astype(int))):
        mask = n_st == sv
        if mask.sum() >= 20:
            ax.boxplot(best_res[mask], positions=[sv], widths=0.6,
                      showfliers=False,
                      boxprops=dict(color='blue'),
                      medianprops=dict(color='red'))
    ax.axhline(y=0, color='gray', lw=0.5)
    ax.set_xlabel('n_adjacent_settlements')
    ax.set_ylabel('residual')
    ax.set_title('Residual by adj settlement count')

    fig.suptitle('Residual analysis on terrain-stable clean data (pooled across rounds)', fontsize=14)
    plt.tight_layout()
    fig.savefig('outlier_nonlinearity.png', dpi=150)
    print("\nSaved outlier_nonlinearity.png")
    plt.close('all')

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Per-round model comparison
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("PER-ROUND MODEL COMPARISON")
    print(f"{'='*72}")
    print(f"{'Round':<15} {'Base R²':>8} {'pop*(1-f)':>10} {'pop+pop²':>10} {'pop+p*f':>10}")

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr2, groups2 = collect_stable_rows(replays)
        pop2 = arr2[:, 0]
        food2 = arr2[:, 1]
        npl2 = arr2[:, 2]
        nfo2 = arr2[:, 3]
        fd2 = arr2[:, 7]
        omf2 = 1 - food2

        # Base model
        feat_base = np.column_stack([npl2*omf2, nfo2*omf2, food2*omf2, pop2, fd2])
        dm_base = np.zeros_like(feat_base)
        for col in range(feat_base.shape[1]):
            dm_base[:, col] = demean_col(feat_base[:, col], groups2)
        _, r2_base, _, _ = fit_ols(dm_base[:, :4], dm_base[:, 4])

        # pop*(1-f) model
        feat_pf = np.column_stack([npl2*omf2, nfo2*omf2, food2*omf2, pop2*omf2, fd2])
        dm_pf = np.zeros_like(feat_pf)
        for col in range(feat_pf.shape[1]):
            dm_pf[:, col] = demean_col(feat_pf[:, col], groups2)
        _, r2_pf, _, _ = fit_ols(dm_pf[:, :4], dm_pf[:, 4])

        # pop + pop² model
        feat_pp = np.column_stack([npl2*omf2, nfo2*omf2, food2*omf2, pop2, pop2**2, fd2])
        dm_pp = np.zeros_like(feat_pp)
        for col in range(feat_pp.shape[1]):
            dm_pp[:, col] = demean_col(feat_pp[:, col], groups2)
        _, r2_pp, _, _ = fit_ols(dm_pp[:, :5], dm_pp[:, 5])

        # pop + pop*food model
        feat_ppf = np.column_stack([npl2*omf2, nfo2*omf2, food2*omf2, pop2, pop2*food2, fd2])
        dm_ppf = np.zeros_like(feat_ppf)
        for col in range(feat_ppf.shape[1]):
            dm_ppf[:, col] = demean_col(feat_ppf[:, col], groups2)
        _, r2_ppf, _, _ = fit_ols(dm_ppf[:, :5], dm_ppf[:, 5])

        print(f"{rid[:14]:<15} {r2_base:>8.4f} {r2_pf:>10.4f} {r2_pp:>10.4f} {r2_ppf:>10.4f}")


if __name__ == "__main__":
    main()
