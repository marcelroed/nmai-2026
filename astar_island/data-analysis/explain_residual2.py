"""
Deeper residual analysis: add n_st*(1-f) and investigate remaining structure.

Build on findings from explain_residual.py:
1. n_st*(1-f) is the biggest missing term
2. Residual vs food shows systematic curve
3. pop×food interaction is significant

Strategy:
- Fit extended model with n_st*(1-f) per round
- Check residual structure after adding n_st
- Try all single-terrain model: all terrain types × (1-f) with their own coefficients
- Check if pop_rate really is a constant or if it depends on food
- Per-round with iterative outlier removal for clean results

Reproduce:  python3 data-analysis/explain_residual2.py
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


def collect_stable(replays):
    """Collect terrain-stable, non-outlier observations with full detail."""
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
                    adj_b.get(EMPTY, 0),
                ))
                groups.append((seed, step))
    return np.array(rows, dtype=np.float64), groups


def iterative_fit(arr, groups, feat_fn, n_iters=3, threshold=3.0):
    """Fit with iterative outlier removal. feat_fn(arr) returns feature columns."""
    mask = np.ones(len(arr), dtype=bool)
    for iteration in range(n_iters + 1):
        a = arr[mask]
        g = [groups[i] for i in range(len(groups)) if mask[i]]
        feats = feat_fn(a)
        dm_X = np.column_stack([demean_col(c, g) for c in feats])
        dm_y = demean_col(a[:, 7], g)  # food_delta is col 7
        coeffs, r2, res = fit_ols(dm_X, dm_y)

        if iteration < n_iters:
            mae = np.mean(np.abs(res))
            keep = np.abs(res) < threshold * mae
            # Update mask
            active = np.where(mask)[0]
            mask[active[~keep]] = False

    return coeffs, r2, res, mask, g


def binned_median(x, y, n_bins=50, x_range=None):
    if x_range is None:
        x_range = (x.min(), x.max())
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    centers, medians = [], []
    for j in range(n_bins):
        m = (x >= bins[j]) & (x < bins[j + 1])
        if m.sum() >= 10:
            centers.append((bins[j] + bins[j + 1]) / 2)
            medians.append(np.median(y[m]))
    return np.array(centers), np.array(medians)


def main():
    rounds = load_replays_by_round()

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Compare model variants per round (with outlier removal)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 90)
    print("PART 1: Model comparison with iterative outlier removal")
    print("=" * 90)

    def base_feats(a):
        f, omf = a[:, 1], 1 - a[:, 1]
        return [a[:, 2]*omf, a[:, 3]*omf, f*omf, a[:, 0]]

    def plus_nst(a):
        f, omf = a[:, 1], 1 - a[:, 1]
        return [a[:, 2]*omf, a[:, 3]*omf, f*omf, a[:, 0], a[:, 6]*omf]

    def plus_nst_pf(a):
        f, omf = a[:, 1], 1 - a[:, 1]
        return [a[:, 2]*omf, a[:, 3]*omf, f*omf, a[:, 0], a[:, 6]*omf, a[:, 0]*f]

    def plus_all_terrain(a):
        """All terrain types × (1-f) + pop."""
        f, omf = a[:, 1], 1 - a[:, 1]
        return [a[:, 2]*omf, a[:, 3]*omf, a[:, 4]*omf, a[:, 5]*omf, a[:, 6]*omf, f*omf, a[:, 0]]

    def plus_all_terrain_pf(a):
        """All terrain + pop*food."""
        f, omf = a[:, 1], 1 - a[:, 1]
        return [a[:, 2]*omf, a[:, 3]*omf, a[:, 4]*omf, a[:, 5]*omf, a[:, 6]*omf, f*omf, a[:, 0], a[:, 0]*f]

    models = [
        ("base (4 feat)",       base_feats,          ["pl", "fo", "fb", "pop"]),
        ("+nst (5 feat)",       plus_nst,            ["pl", "fo", "fb", "pop", "st"]),
        ("+nst+p*f (6 feat)",   plus_nst_pf,         ["pl", "fo", "fb", "pop", "st", "p*f"]),
        ("all_terrain (7)",     plus_all_terrain,     ["pl", "fo", "mt", "oc", "st", "fb", "pop"]),
        ("all_terrain+p*f (8)", plus_all_terrain_pf,  ["pl", "fo", "mt", "oc", "st", "fb", "pop", "p*f"]),
    ]

    header = f"{'Round':<14} " + " ".join(f"{m[0]:>20s}" for m in models)
    print(header)
    print("-" * len(header))

    all_coeffs = {name: [] for name, _, _ in models}

    for rid in sorted(rounds):
        arr, groups = collect_stable(rounds[rid])
        line = f"{rid[:13]:<14}"
        for name, feat_fn, labels in models:
            coeffs, r2, res, mask, g = iterative_fit(arr, groups, feat_fn)
            line += f" {r2:>20.4f}"
            all_coeffs[name].append(coeffs)
        print(line)

    # Print coefficients for key models
    print(f"\n{'='*90}")
    print("COEFFICIENTS (per-round, after outlier removal)")
    print(f"{'='*90}")
    for name, _, labels in models:
        print(f"\n  {name}:")
        coeff_arr = np.array(all_coeffs[name])
        for i, lab in enumerate(labels):
            vals = coeff_arr[:, i]
            print(f"    {lab:>5s}: mean={vals.mean():.6f} ± {vals.std():.6f}  "
                  f"range=[{vals.min():.6f}, {vals.max():.6f}]")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Residual structure AFTER adding n_st
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("PART 2: Residual analysis of extended model (+n_st)")
    print(f"{'='*90}")

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    pooled_res = []
    pooled_food = []
    pooled_pop = []
    pooled_nst = []
    pooled_npl = []
    pooled_nfo = []
    pooled_nmt = []
    pooled_noc = []
    pooled_omf = []
    pooled_defense = []

    for idx, rid in enumerate(sorted(rounds)):
        arr, groups = collect_stable(rounds[rid])

        # Fit extended model with outlier removal
        coeffs, r2, res, mask, g_clean = iterative_fit(arr, groups, plus_nst)
        arr_clean = arr[mask]

        # Compute residual on cleaned data
        f = arr_clean[:, 1]
        omf = 1 - f
        feats = plus_nst(arr_clean)
        dm_X = np.column_stack([demean_col(c, g_clean) for c in feats])
        dm_y = demean_col(arr_clean[:, 7], g_clean)
        pred = dm_X @ coeffs
        res = dm_y - pred

        pooled_res.append(res)
        pooled_food.append(f)
        pooled_pop.append(arr_clean[:, 0])
        pooled_nst.append(arr_clean[:, 6])
        pooled_npl.append(arr_clean[:, 2])
        pooled_nfo.append(arr_clean[:, 3])
        pooled_nmt.append(arr_clean[:, 4])
        pooled_noc.append(arr_clean[:, 5])
        pooled_omf.append(omf)
        pooled_defense.append(arr_clean[:, 10])

        # Plot residual vs food
        ax = axes[idx]
        ax.scatter(f, res, s=1, alpha=0.05, color="steelblue", rasterized=True)
        cx, my = binned_median(f, res, 40, (0, 1))
        ax.plot(cx, my, "r-", lw=2, label="median (+nst model)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel("food")
        ax.set_ylabel("residual")
        ax.set_title(f"{rid[:8]}  R²={r2:.4f}")
        ax.legend(fontsize=7)

    fig.suptitle("Residual vs food AFTER adding n_st*(1-f) to model", fontsize=14)
    plt.tight_layout()
    fig.savefig("residual2_vs_food.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved residual2_vs_food.png")

    # Pool and analyze
    res = np.concatenate(pooled_res)
    food = np.concatenate(pooled_food)
    pop = np.concatenate(pooled_pop)
    nst = np.concatenate(pooled_nst)
    npl = np.concatenate(pooled_npl)
    nfo = np.concatenate(pooled_nfo)
    nmt = np.concatenate(pooled_nmt)
    noc = np.concatenate(pooled_noc)
    omf = np.concatenate(pooled_omf)
    defense = np.concatenate(pooled_defense)

    print(f"\nPooled (cleaned): {len(res)} observations")
    print(f"MAE: {np.mean(np.abs(res)):.6f}")

    # Remaining single-feature correlations
    candidates = [
        ("food",       food),
        ("food²",      food**2),
        ("pop",        pop),
        ("pop²",       pop**2),
        ("pop*food",   pop * food),
        ("pop*(1-f)",  pop * omf),
        ("n_st",       nst),
        ("n_pl",       npl),
        ("n_fo",       nfo),
        ("defense",    defense),
        ("f²*(1-f)",   food**2 * omf),
        ("f*(1-f)²",   food * omf**2),
        ("(1-f)²",     omf**2),
    ]

    print(f"\n  Remaining correlations with residual:")
    print(f"  {'Feature':<12} {'corr':>10} {'R²':>10}")
    print(f"  {'─'*35}")
    for name, feat in candidates:
        corr = np.corrcoef(res, feat)[0, 1]
        X = feat.reshape(-1, 1)
        c, r2, _ = fit_ols(X, res)
        print(f"  {name:<12} {corr:>10.6f} {r2:>10.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Is pop really linear, or does it depend on food?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("PART 3: Pop effect — linear vs food-dependent")
    print("  If pop_rate depends on food, we'd see residual = f(pop, food)")
    print(f"{'='*90}")

    # Fit interaction model per round
    for rid in sorted(rounds):
        arr, groups = collect_stable(rounds[rid])

        # Model A: base + nst  (pop is linear)
        _, r2_a, _, mask_a, _ = iterative_fit(arr, groups, plus_nst)

        # Model B: base + nst + pop*food
        _, r2_b, _, mask_b, _ = iterative_fit(arr, groups, plus_nst_pf)

        print(f"  {rid[:13]}: +nst R²={r2_a:.4f}  +nst+p*f R²={r2_b:.4f}  Δ={r2_b-r2_a:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: What are the actual residual "hotspots"?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("PART 4: Residual hotspots — where is |residual| large?")
    print(f"{'='*90}")

    # For the +nst model, examine the 5% worst residuals
    threshold = np.percentile(np.abs(res), 95)
    worst = np.abs(res) > threshold
    good = np.abs(res) <= threshold

    print(f"\n  {worst.sum()} worst observations (top 5%), |res| > {threshold:.4f}")
    print(f"  Characteristic comparison (worst vs rest):")
    for name, vals in [("food", food), ("pop", pop), ("n_st", nst),
                       ("n_pl", npl), ("n_fo", nfo), ("defense", defense)]:
        print(f"    {name:<10s}: worst={vals[worst].mean():.3f}  "
              f"rest={vals[good].mean():.3f}")

    # Check if worst residuals cluster at specific food values
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

    vars_plot = [
        ("food", food, (0, 1)),
        ("pop", pop, (0, 3)),
        ("n_settlement", nst, (-0.5, 5.5)),
        ("n_plains", npl, (-0.5, 8.5)),
        ("defense", defense, (0, 1)),
        ("food*(1-food)", food*omf, (0, 0.26)),
    ]

    for i, (name, var, xr) in enumerate(vars_plot):
        ax = axes2.flatten()[i]
        ax.scatter(var, res, s=0.5, alpha=0.03, color="steelblue", rasterized=True)
        cx, my = binned_median(var, res, 40, xr)
        ax.plot(cx, my, "r-", lw=2.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(*xr)
        ax.set_ylim(-0.06, 0.06)
        ax.set_xlabel(name)
        ax.set_ylabel("residual")
        ax.set_title(f"Residual vs {name} (+nst model)")

    fig2.suptitle("Residual structure after adding n_st*(1-f)", fontsize=14)
    plt.tight_layout()
    fig2.savefig("residual2_vs_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved residual2_vs_all.png")

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: Could it be rounding / quantization noise?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("PART 5: Quantization analysis")
    print(f"{'='*90}")

    # Check if residual is quantized
    # If the game computes food_delta at integer precision (multiples of 0.001),
    # the residual should show quantized structure
    res_mod = np.abs(res * 1000 - np.round(res * 1000))
    print(f"\n  Residual quantization to 0.001:")
    print(f"    mean |res*1000 - round(res*1000)| = {res_mod.mean():.6f}")
    print(f"    % within 0.0001 of 0.001 grid: {100*np.mean(res_mod < 0.1):.1f}%")

    # Histogram of residuals
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(res, bins=200, range=(-0.1, 0.1), color="steelblue", alpha=0.7)
    ax1.set_xlabel("residual")
    ax1.set_ylabel("count")
    ax1.set_title("Residual distribution (+nst model)")
    ax1.axvline(0, color="red", lw=1)

    # Fine-grained: are there spikes at multiples of some value?
    ax2.hist(res, bins=2000, range=(-0.05, 0.05), color="steelblue", alpha=0.7)
    ax2.set_xlabel("residual")
    ax2.set_ylabel("count")
    ax2.set_title("Residual distribution (zoomed)")
    ax2.axvline(0, color="red", lw=1)

    plt.tight_layout()
    fig3.savefig("residual2_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved residual2_histogram.png")

    # ═══════════════════════════════════════════════════════════════════
    # PART 6: Check if the food-dependent pattern is a functional form issue
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("PART 6: Alternative functional forms for food feedback")
    print(f"{'='*90}")

    for rid in sorted(rounds)[:3]:
        arr, groups = collect_stable(rounds[rid])

        f = arr[:, 1]
        omf = 1 - f

        def model_base(a):
            f, omf = a[:, 1], 1 - a[:, 1]
            return [a[:, 2]*omf, a[:, 3]*omf, f*omf, a[:, 0], a[:, 6]*omf]

        def model_f2(a):
            """Try f²*(1-f) instead of f*(1-f)."""
            f, omf = a[:, 1], 1 - a[:, 1]
            return [a[:, 2]*omf, a[:, 3]*omf, f**2*omf, a[:, 0], a[:, 6]*omf]

        def model_f_omf2(a):
            """Try f*(1-f)² instead of f*(1-f)."""
            f, omf = a[:, 1], 1 - a[:, 1]
            return [a[:, 2]*omf, a[:, 3]*omf, f*omf**2, a[:, 0], a[:, 6]*omf]

        def model_f_and_f2(a):
            """Both f*(1-f) and f²*(1-f)."""
            f, omf = a[:, 1], 1 - a[:, 1]
            return [a[:, 2]*omf, a[:, 3]*omf, f*omf, f**2*omf, a[:, 0], a[:, 6]*omf]

        def model_no_feedback(a):
            """No food feedback, just terrain*(1-f)."""
            f, omf = a[:, 1], 1 - a[:, 1]
            return [a[:, 2]*omf, a[:, 3]*omf, a[:, 0], a[:, 6]*omf]

        def model_pop_omf(a):
            """pop*(1-f) instead of pop."""
            f, omf = a[:, 1], 1 - a[:, 1]
            return [a[:, 2]*omf, a[:, 3]*omf, f*omf, a[:, 0]*omf, a[:, 6]*omf]

        variants = [
            ("f*(1-f)",       model_base),
            ("f²*(1-f)",      model_f2),
            ("f*(1-f)²",      model_f_omf2),
            ("f*(1-f)+f²*(1-f)", model_f_and_f2),
            ("no feedback",   model_no_feedback),
            ("pop*(1-f)",     model_pop_omf),
        ]

        print(f"\n  {rid[:12]}:")
        for name, fn in variants:
            _, r2, _, _, _ = iterative_fit(arr, groups, fn)
            print(f"    {name:<22s}: R²={r2:.4f}")


if __name__ == "__main__":
    main()
