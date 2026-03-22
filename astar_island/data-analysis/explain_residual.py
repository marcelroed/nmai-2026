"""
Investigate what the residual from the food formula still contains.

Current model (R²=0.91-0.98 after cleaning):
  food_delta = pl*(1-f) + fo*(1-f) + fb*f*(1-f) - pop_rate*pop + weather

Questions:
1. Is the residual structured (vs food, pop, terrain, step)?
2. Does the model form need correction?
3. Are there additional terms we're missing?

Reproduce:  python3 data-analysis/explain_residual.py
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
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def demean_col(vals, groups):
    gmap = defaultdict(list)
    for i, g in enumerate(groups):
        gmap[g].append(i)
    out = vals.copy()
    for indices in gmap.values():
        out[indices] -= vals[indices].mean()
    return out


def collect_stable(replays):
    rows = []
    groups = []
    for replay in replays:
        frames = replay["frames"]
        seed = replay["seed_index"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            grid_b, grid_a = fb["grid"], fa["grid"]
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
                if grid_b[y][x] == PORT:
                    continue
                if sa["defense"] - sb["defense"] < -0.001:
                    continue
                if sa["food"] >= 0.997 or sa["food"] <= 0.001:
                    continue
                if sa["population"] - sb["population"] < -0.05:
                    continue
                adj_b = terrain_adj(grid_b, x, y)
                adj_a = terrain_adj(grid_a, x, y)
                if dict(adj_b) != dict(adj_a):
                    continue
                rows.append((
                    sb["population"], sb["food"],
                    adj_b.get(PLAINS, 0), adj_b.get(FOREST, 0),
                    adj_b.get(MOUNTAIN, 0), adj_b.get(OCEAN, 0),
                    adj_b.get(SETTLEMENT, 0) + adj_b.get(PORT, 0) + adj_b.get(RUIN, 0),
                    sa["food"] - sb["food"],
                    step, seed,
                    sb["defense"], sb["wealth"],
                    sa["population"] - sb["population"],
                ))
                groups.append((seed, step))
    return np.array(rows, dtype=np.float64), groups


def binned_median(x, y, n_bins=50, x_range=None):
    if x_range is None:
        x_range = (x.min(), x.max())
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    centers, medians, counts = [], [], []
    for j in range(n_bins):
        mask = (x >= bins[j]) & (x < bins[j + 1])
        c = mask.sum()
        if c >= 10:
            centers.append((bins[j] + bins[j + 1]) / 2)
            medians.append(np.median(y[mask]))
            counts.append(c)
    return np.array(centers), np.array(medians), np.array(counts)


def main():
    rounds = load_replays_by_round()

    # Pool all rounds, fit per-round, collect residuals
    all_res = []
    all_food = []
    all_pop = []
    all_npl = []
    all_nfo = []
    all_step = []
    all_defense = []
    all_wealth = []
    all_pop_delta = []
    all_round = []
    all_omf = []
    all_nmt = []
    all_noc = []
    all_nst = []

    fig1, axes1 = plt.subplots(3, 3, figsize=(20, 16))
    axes1 = axes1.flatten()

    for idx, rid in enumerate(sorted(rounds)):
        arr, groups = collect_stable(rounds[rid])

        pop, food = arr[:, 0], arr[:, 1]
        npl, nfo = arr[:, 2], arr[:, 3]
        nmt, noc, nst = arr[:, 4], arr[:, 5], arr[:, 6]
        fd = arr[:, 7]
        step = arr[:, 8]
        omf = 1 - food

        # Fit the model
        feat = [npl * omf, nfo * omf, food * omf, pop]
        dm_X = np.column_stack([demean_col(c, groups) for c in feat])
        dm_y = demean_col(fd, groups)
        coeffs, r2, mae, res = fit_ols(dm_X, dm_y)

        all_res.append(res)
        all_food.append(food)
        all_pop.append(pop)
        all_npl.append(npl)
        all_nfo.append(nfo)
        all_nmt.append(nmt)
        all_noc.append(noc)
        all_nst.append(nst)
        all_step.append(step)
        all_defense.append(arr[:, 10])
        all_wealth.append(arr[:, 11])
        all_pop_delta.append(arr[:, 12])
        all_round.append(np.full(len(arr), idx))
        all_omf.append(omf)

        # Per-round: residual vs food with median curve
        ax = axes1[idx]
        ax.scatter(food, res, s=1, alpha=0.05, color="steelblue", rasterized=True)
        cx, my, _ = binned_median(food, res, 40, (0, 1))
        ax.plot(cx, my, "r-", lw=2, label="median")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel("food")
        ax.set_ylabel("residual")
        ax.set_title(f"{rid[:8]}  R²={r2:.3f}")
        ax.legend(fontsize=7)

    fig1.suptitle("Residual vs food — is there a systematic curve?", fontsize=14)
    plt.tight_layout()
    fig1.savefig("residual_vs_food.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved residual_vs_food.png")

    # Pool everything
    res = np.concatenate(all_res)
    food = np.concatenate(all_food)
    pop = np.concatenate(all_pop)
    npl = np.concatenate(all_npl)
    nfo = np.concatenate(all_nfo)
    nmt = np.concatenate(all_nmt)
    noc = np.concatenate(all_noc)
    nst = np.concatenate(all_nst)
    step = np.concatenate(all_step)
    defense = np.concatenate(all_defense)
    wealth = np.concatenate(all_wealth)
    pop_delta = np.concatenate(all_pop_delta)
    omf = np.concatenate(all_omf)
    round_idx = np.concatenate(all_round)

    print(f"\nTotal pooled observations: {len(res)}")
    print(f"Overall MAE: {np.mean(np.abs(res)):.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # Figure 2: Residual vs every variable (pooled, median curves)
    # ═══════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(3, 4, figsize=(22, 14))

    variables = [
        ("food", food, (0, 1)),
        ("population", pop, (0, 4)),
        ("n_plains", npl, (-0.5, 8.5)),
        ("n_forest", nfo, (-0.5, 8.5)),
        ("n_mountain", nmt, (-0.5, 5.5)),
        ("n_ocean", noc, (-0.5, 5.5)),
        ("n_settlement", nst, (-0.5, 5.5)),
        ("step", step, (0, 50)),
        ("defense", defense, (0, 1)),
        ("wealth", wealth, (0, 0.5)),
        ("pop_delta", pop_delta, (-0.05, 0.5)),
        ("food*(1-food)", food * omf, (0, 0.25)),
    ]

    for i, (name, var, xr) in enumerate(variables):
        ax = axes2.flatten()[i]
        ax.scatter(var, res, s=0.5, alpha=0.03, color="steelblue", rasterized=True)
        cx, my, _ = binned_median(var, res, 40, xr)
        ax.plot(cx, my, "r-", lw=2.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(*xr)
        ax.set_ylim(-0.08, 0.08)
        ax.set_xlabel(name)
        ax.set_ylabel("residual")
        ax.set_title(name)

    fig2.suptitle("Residual vs every variable (pooled, median curve in red)", fontsize=14)
    plt.tight_layout()
    fig2.savefig("residual_vs_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved residual_vs_all.png")

    # ═══════════════════════════════════════════════════════════════════
    # Quantify: what additional terms reduce residual?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Residual regression: what predicts the residual?")
    print(f"{'='*70}")

    # Try regressing residual on candidate features (no demeaning needed,
    # residual is already demeaned)
    candidates = [
        ("food",        food),
        ("food²",       food**2),
        ("pop²",        pop**2),
        ("pop*food",    pop * food),
        ("pop*(1-f)",   pop * omf),
        ("n_mt*(1-f)",  nmt * omf),
        ("n_oc*(1-f)",  noc * omf),
        ("n_st*(1-f)",  nst * omf),
        ("defense",     defense),
        ("wealth",      wealth),
        ("pop_delta",   pop_delta),
        ("step",        step),
        ("f²*(1-f)",    food**2 * omf),
        ("f*(1-f)²",    food * omf**2),
        ("(1-f)²",      omf**2),
    ]

    print(f"\n  {'Feature':<16} {'corr(res,feat)':>15} {'R² if added':>12}")
    print("  " + "─" * 45)
    for name, feat in candidates:
        corr = np.corrcoef(res, feat)[0, 1]
        # R² of regressing residual on this feature
        X = feat.reshape(-1, 1)
        c, r2_add, _, _ = fit_ols(X, res)
        print(f"  {name:<16} {corr:>15.6f} {r2_add:>12.6f}")

    # Multi-feature: which combination explains the most residual?
    print(f"\n  Multi-feature residual models:")
    combos = [
        ("pop*food",                 [pop * food]),
        ("pop*food + pop²",          [pop * food, pop**2]),
        ("n_st*(1-f)",               [nst * omf]),
        ("n_mt*(1-f) + n_oc*(1-f)",  [nmt * omf, noc * omf]),
        ("pop*food + n_st*(1-f)",    [pop * food, nst * omf]),
        ("food² + pop*food",         [food**2, pop * food]),
        ("f²*(1-f) + pop*food",      [food**2 * omf, pop * food]),
        ("pop*food + defense",       [pop * food, defense]),
        ("pop*food + wealth",        [pop * food, wealth]),
        ("all numeric",              [pop*food, pop**2, nmt*omf, noc*omf, nst*omf,
                                      defense, wealth, pop_delta]),
    ]
    for name, feats in combos:
        X = np.column_stack(feats)
        _, r2, mae_c, _ = fit_ols(X, res)
        print(f"  {name:<35} R²={r2:.6f}  MAE={mae_c:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # Figure 3: Residual vs food, split by pop level
    # ═══════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes3[0]
    pop_bins = [(0, 0.6, "blue"), (0.6, 1.0, "green"),
                (1.0, 1.5, "orange"), (1.5, 5.0, "red")]
    for lo, hi, col in pop_bins:
        mask = (pop >= lo) & (pop < hi)
        if mask.sum() > 100:
            cx, my, _ = binned_median(food[mask], res[mask], 40, (0, 1))
            ax.plot(cx, my, ".-", color=col, lw=1.5,
                    label=f"pop=[{lo},{hi}) n={mask.sum()}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("food")
    ax.set_ylabel("median residual")
    ax.set_title("Residual vs food, split by pop")
    ax.set_ylim(-0.08, 0.08)
    ax.legend(fontsize=8)

    ax = axes3[1]
    food_bins = [(0, 0.3, "purple"), (0.3, 0.6, "blue"),
                 (0.6, 0.8, "green"), (0.8, 1.0, "orange")]
    for lo, hi, col in food_bins:
        mask = (food >= lo) & (food < hi)
        if mask.sum() > 100:
            cx, my, _ = binned_median(pop[mask], res[mask], 30, (0, 3))
            ax.plot(cx, my, ".-", color=col, lw=1.5,
                    label=f"food=[{lo},{hi}) n={mask.sum()}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("population")
    ax.set_ylabel("median residual")
    ax.set_title("Residual vs pop, split by food")
    ax.set_ylim(-0.08, 0.08)
    ax.legend(fontsize=8)

    fig3.suptitle("Residual interaction: food × pop", fontsize=14)
    plt.tight_layout()
    fig3.savefig("residual_interaction.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("\nSaved residual_interaction.png")

    # ═══════════════════════════════════════════════════════════════════
    # Try the improved model on each round
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Per-round: base model vs +pop*food vs +pop*food+n_st*(1-f)")
    print(f"{'='*70}")
    print(f"{'Round':<14} {'Base':>8} {'+p*f':>8} {'+p*f+st':>9} {'+p*f+st+mt+oc':>15}")

    for rid in sorted(rounds):
        arr, groups = collect_stable(rounds[rid])
        p, f = arr[:, 0], arr[:, 1]
        np_, nf_ = arr[:, 2], arr[:, 3]
        nm_, no_, ns_ = arr[:, 4], arr[:, 5], arr[:, 6]
        fd = arr[:, 7]
        of_ = 1 - f

        def dm_fit(feat_list):
            dm_X = np.column_stack([demean_col(c, groups) for c in feat_list])
            dm_y = demean_col(fd, groups)
            _, r2, _, _ = fit_ols(dm_X, dm_y)
            return r2

        base = [np_*of_, nf_*of_, f*of_, p]
        r2_base = dm_fit(base)
        r2_pf = dm_fit(base + [p * f])
        r2_pf_st = dm_fit(base + [p * f, ns_ * of_])
        r2_all = dm_fit(base + [p * f, ns_ * of_, nm_ * of_, no_ * of_])

        print(f"{rid[:13]:<14} {r2_base:>8.4f} {r2_pf:>8.4f} {r2_pf_st:>9.4f} {r2_all:>15.4f}")


if __name__ == "__main__":
    main()
