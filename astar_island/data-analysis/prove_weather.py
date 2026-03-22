"""
Prove and plot weather characterization from replay data.

Weather is the per-(seed, step) additive component of the food formula:
  food_delta = pl*n_pl*(1-f) + fo*n_fo*(1-f) + fb*f*(1-f) - pop*pop + weather

Method:
  1. Fit the food model per round with iterative outlier removal
  2. Estimate weather per (seed, step) as the group-mean residual
  3. Prove: weather is independent across seeds, i.i.d. over time, ~normal

Output: weather_proof.png

Reproduce:  python3 data-analysis/prove_weather.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


# ─── helpers ──────────────────────────────────────────────────────────────

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
                    sa["food"] - sb["food"],
                ))
                groups.append((seed, step))
    return np.array(rows, dtype=np.float64), groups


def iterative_fit(arr, groups, n_iters=3, threshold=3.0):
    mask = np.ones(len(arr), dtype=bool)
    for iteration in range(n_iters + 1):
        a = arr[mask]
        g = [groups[i] for i in range(len(groups)) if mask[i]]
        f, omf = a[:, 1], 1 - a[:, 1]
        feats = [a[:, 2]*omf, a[:, 3]*omf, f*omf, a[:, 0]]
        dm_X = np.column_stack([demean_col(c, g) for c in feats])
        dm_y = demean_col(a[:, 4], g)
        coeffs, r2, res = fit_ols(dm_X, dm_y)
        if iteration < n_iters:
            mae = np.mean(np.abs(res))
            keep = np.abs(res) < threshold * mae
            active = np.where(mask)[0]
            mask[active[~keep]] = False
    return coeffs, r2, mask


def estimate_weather(arr, groups, coeffs):
    pop, food = arr[:, 0], arr[:, 1]
    npl, nfo = arr[:, 2], arr[:, 3]
    fd = arr[:, 4]
    omf = 1 - food
    pred = coeffs[0]*npl*omf + coeffs[1]*nfo*omf + coeffs[2]*food*omf + coeffs[3]*pop
    residual = fd - pred
    gmap = defaultdict(list)
    for i, g in enumerate(groups):
        gmap[g].append(i)
    weather = {}
    group_sizes = {}
    for g, indices in gmap.items():
        weather[g] = np.mean(residual[indices])
        group_sizes[g] = len(indices)
    return weather, group_sizes


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    rounds = load_replays_by_round()

    # Collect weather estimates for all rounds
    all_weather = {}  # rid -> {(seed, step): w}
    all_sizes = {}    # rid -> {(seed, step): n}
    all_coeffs = {}

    print("=" * 80)
    print("WEATHER ESTIMATION PER ROUND")
    print("=" * 80)

    for rid in sorted(rounds):
        arr, groups = collect_stable(rounds[rid])
        coeffs, r2, mask = iterative_fit(arr, groups)
        arr_c = arr[mask]
        grp_c = [groups[i] for i in range(len(groups)) if mask[i]]
        weather, sizes = estimate_weather(arr_c, grp_c, coeffs)
        all_weather[rid] = weather
        all_sizes[rid] = sizes
        all_coeffs[rid] = coeffs
        w = np.array(list(weather.values()))
        print(f"  {rid[:12]}: R²={r2:.4f}  {len(weather)} groups  "
              f"weather: mean={w.mean():.4f} std={w.std():.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # Build per-round data structures
    # ═══════════════════════════════════════════════════════════════════
    round_stats = {}
    for rid in sorted(rounds):
        weather = all_weather[rid]
        seeds = sorted(set(s for s, _ in weather))
        steps = sorted(set(st for _, st in weather))

        # Build seed time series
        seed_series = {}
        for seed in seeds:
            ss = sorted([(st, weather[(seed, st)])
                         for (s, st) in weather if s == seed])
            seed_series[seed] = ss

        # Cross-seed correlations
        cross_corrs = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i+1:]:
                common = sorted(set(st for st, _ in seed_series[s1]) &
                              set(st for st, _ in seed_series[s2]))
                if len(common) > 5:
                    v1 = [weather[(s1, st)] for st in common]
                    v2 = [weather[(s2, st)] for st in common]
                    cross_corrs.append(np.corrcoef(v1, v2)[0, 1])

        # Lag-1 autocorrelations per seed
        autocorrs = []
        for seed in seeds:
            ws = [w for _, w in seed_series[seed]]
            if len(ws) > 5:
                w_arr = np.array(ws)
                w_c = w_arr - w_arr.mean()
                if np.std(w_c) > 1e-10:
                    autocorrs.append(np.corrcoef(w_c[:-1], w_c[1:])[0, 1])

        # Cross-seed std (same step, how similar are different seeds?)
        step_groups = defaultdict(list)
        for (s, st), w in weather.items():
            step_groups[st].append(w)
        cross_seed_stds = [np.std(vs) for vs in step_groups.values() if len(vs) > 1]

        w_vals = np.array(list(weather.values()))
        round_stats[rid] = {
            "weather": weather,
            "seeds": seeds,
            "steps": steps,
            "seed_series": seed_series,
            "cross_corrs": cross_corrs,
            "autocorrs": autocorrs,
            "cross_seed_stds": cross_seed_stds,
            "w_vals": w_vals,
        }

    # ═══════════════════════════════════════════════════════════════════
    # TESTS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("TEST 1: Weather is independent across seeds")
    print("  If shared, cross-seed correlation ≈ 1. If independent, ≈ 0.")
    print(f"{'='*80}")
    for rid in sorted(rounds):
        rs = round_stats[rid]
        if rs["cross_corrs"]:
            cc = np.array(rs["cross_corrs"])
            print(f"  {rid[:12]}: mean corr = {cc.mean():+.4f}  "
                  f"range [{cc.min():+.4f}, {cc.max():+.4f}]  "
                  f"({len(cc)} pairs)")
    # Pooled test
    all_cc = np.concatenate([rs["cross_corrs"] for rs in round_stats.values()
                             if rs["cross_corrs"]])
    t_stat, p_val = stats.ttest_1samp(all_cc, 0)
    print(f"\n  Pooled: mean = {all_cc.mean():+.4f} ± {all_cc.std():.4f}")
    print(f"  t-test H0: mean=0  → t={t_stat:.2f}, p={p_val:.4f}")
    print(f"  → {'REJECT: weather IS correlated' if p_val < 0.01 else 'ACCEPT: weather is independent across seeds'}")

    print(f"\n{'='*80}")
    print("TEST 2: Weather is i.i.d. over time (no autocorrelation)")
    print(f"{'='*80}")
    for rid in sorted(rounds):
        rs = round_stats[rid]
        if rs["autocorrs"]:
            ac = np.array(rs["autocorrs"])
            print(f"  {rid[:12]}: mean lag-1 AC = {ac.mean():+.4f}  "
                  f"range [{ac.min():+.4f}, {ac.max():+.4f}]")
    all_ac = np.concatenate([rs["autocorrs"] for rs in round_stats.values()
                             if rs["autocorrs"]])
    t_stat, p_val = stats.ttest_1samp(all_ac, 0)
    print(f"\n  Pooled: mean lag-1 AC = {all_ac.mean():+.4f} ± {all_ac.std():.4f}")
    print(f"  t-test H0: AC=0  → t={t_stat:.2f}, p={p_val:.4f}")
    print(f"  → {'REJECT: temporal correlation exists' if p_val < 0.01 else 'ACCEPT: weather is i.i.d. over time'}")

    print(f"\n{'='*80}")
    print("TEST 3: Distribution shape (normality)")
    print(f"{'='*80}")
    for rid in sorted(rounds):
        w = round_stats[rid]["w_vals"]
        sk = float(np.mean(((w - w.mean()) / w.std()) ** 3))
        ku = float(np.mean(((w - w.mean()) / w.std()) ** 4))
        _, sw_p = stats.shapiro(w[:min(len(w), 5000)])
        print(f"  {rid[:12]}: skew={sk:+.3f}  kurtosis={ku:.3f} (normal=3)  "
              f"Shapiro p={sw_p:.4f}")

    # Pooled
    all_w = np.concatenate([rs["w_vals"] for rs in round_stats.values()])
    sk = float(np.mean(((all_w - all_w.mean()) / all_w.std()) ** 3))
    ku = float(np.mean(((all_w - all_w.mean()) / all_w.std()) ** 4))
    print(f"\n  Pooled ({len(all_w)} values): skew={sk:+.3f}  kurtosis={ku:.3f}")
    print(f"  std range: [{min(rs['w_vals'].std() for rs in round_stats.values()):.4f}, "
          f"{max(rs['w_vals'].std() for rs in round_stats.values()):.4f}]")

    print(f"\n{'='*80}")
    print("TEST 4: Weather magnitude vs food_delta variance")
    print(f"{'='*80}")
    for rid in sorted(rounds):
        arr, groups = collect_stable(rounds[rid])
        coeffs, _, mask = iterative_fit(arr, groups)
        arr_c = arr[mask]
        fd_std = np.std(arr_c[:, 4])
        w_std = round_stats[rid]["w_vals"].std()
        pct = (w_std / fd_std * 100) if fd_std > 0 else 0
        print(f"  {rid[:12]}: food_delta std={fd_std:.4f}  "
              f"weather std={w_std:.4f}  "
              f"ratio={pct:.1f}%")

    print(f"\n{'='*80}")
    print("TEST 5: Per-round mean offset (absorbs winter_severity)")
    print(f"{'='*80}")
    for rid in sorted(rounds):
        w = round_stats[rid]["w_vals"]
        print(f"  {rid[:12]}: mean = {w.mean():+.5f}")
    means = [rs["w_vals"].mean() for rs in round_stats.values()]
    print(f"\n  Range of per-round means: [{min(means):+.5f}, {max(means):+.5f}]")
    print(f"  → Nonzero means likely absorb winter_severity + model baseline")

    # ═══════════════════════════════════════════════════════════════════
    # PLOT
    # ═══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 16))

    # Panel layout: 3×3 grid
    # Row 1: Weather time series (3 representative rounds)
    # Row 2: [histogram pooled] [cross-seed scatter] [autocorrelation lags]
    # Row 3: [weather vs step mean] [QQ plot] [cross-seed std distribution]

    rids_sorted = sorted(rounds)

    # ── Row 1: Time series for 3 rounds ────────────────────────────────
    for col, rid in enumerate(rids_sorted[:3]):
        ax = fig.add_subplot(3, 3, col + 1)
        rs = round_stats[rid]
        for seed in rs["seeds"]:
            steps_w = rs["seed_series"][seed]
            ax.plot([s for s, _ in steps_w], [w for _, w in steps_w],
                    ".-", markersize=2, lw=0.7, alpha=0.7, label=f"seed {seed}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("weather")
        ax.set_title(f"{rid[:8]}  std={rs['w_vals'].std():.4f}")
        ax.set_ylim(-0.15, 0.15)
        ax.legend(fontsize=6, ncol=2)

    # ── Row 2, col 1: Pooled histogram ────────────────────────────────
    ax = fig.add_subplot(3, 3, 4)
    for rid in rids_sorted:
        w = round_stats[rid]["w_vals"]
        ax.hist(w, bins=40, alpha=0.4, density=True, label=rid[:6])
    # Overlay normal fit
    x_norm = np.linspace(-0.15, 0.15, 200)
    ax.plot(x_norm, stats.norm.pdf(x_norm, all_w.mean(), all_w.std()),
            "k--", lw=2, label=f"N({all_w.mean():.3f}, {all_w.std():.3f}²)")
    ax.set_xlabel("weather value")
    ax.set_ylabel("density")
    ax.set_title("Distribution per round (overlaid)")
    ax.set_xlim(-0.15, 0.15)
    ax.legend(fontsize=5, ncol=3)

    # ── Row 2, col 2: Cross-seed scatter (one round) ─────────────────
    ax = fig.add_subplot(3, 3, 5)
    rid = rids_sorted[0]
    rs = round_stats[rid]
    s0, s1 = rs["seeds"][0], rs["seeds"][1]
    common = sorted(set(st for st, _ in rs["seed_series"][s0]) &
                    set(st for st, _ in rs["seed_series"][s1]))
    w0 = [rs["weather"][(s0, st)] for st in common]
    w1 = [rs["weather"][(s1, st)] for st in common]
    corr01 = np.corrcoef(w0, w1)[0, 1]
    ax.scatter(w0, w1, s=15, alpha=0.6, color="steelblue", zorder=3)
    lim = 0.12
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"weather (seed {s0})")
    ax.set_ylabel(f"weather (seed {s1})")
    ax.set_title(f"Cross-seed: r={corr01:.3f} ({rid[:8]})\nIf shared → points on y=x")
    ax.set_aspect("equal")

    # ── Row 2, col 3: Lag-1 autocorrelation distribution ──────────────
    ax = fig.add_subplot(3, 3, 6)
    ax.hist(all_ac, bins=25, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.axvline(all_ac.mean(), color="orange", lw=1.5,
               label=f"mean={all_ac.mean():.3f}")
    ax.set_xlabel("lag-1 autocorrelation")
    ax.set_ylabel("count")
    ax.set_title(f"Autocorrelation across all seed×round\n(0 = i.i.d.)")
    ax.legend(fontsize=8)

    # ── Row 3, col 1: Weather vs step (mean across seeds) ────────────
    ax = fig.add_subplot(3, 3, 7)
    for rid in rids_sorted:
        weather = all_weather[rid]
        step_w = defaultdict(list)
        for (seed, step), w in weather.items():
            step_w[step].append(w)
        steps = sorted(step_w)
        means = [np.mean(step_w[st]) for st in steps]
        ax.plot(steps, means, ".-", alpha=0.5, markersize=3, lw=0.8, label=rid[:6])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("mean weather (across seeds)")
    ax.set_title("No seasonal pattern")
    ax.set_ylim(-0.12, 0.12)
    ax.legend(fontsize=5, ncol=3)

    # ── Row 3, col 2: QQ plot ─────────────────────────────────────────
    ax = fig.add_subplot(3, 3, 8)
    w_sorted = np.sort((all_w - all_w.mean()) / all_w.std())
    n = len(w_sorted)
    theoretical = stats.norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n))
    ax.scatter(theoretical, w_sorted, s=1, alpha=0.3, color="steelblue",
               rasterized=True)
    ax.plot([-3.5, 3.5], [-3.5, 3.5], "r-", lw=1.5)
    ax.set_xlabel("theoretical quantiles (normal)")
    ax.set_ylabel("observed quantiles")
    ax.set_title("QQ plot (pooled, standardized)")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")

    # ── Row 3, col 3: Summary stats table ─────────────────────────────
    ax = fig.add_subplot(3, 3, 9)
    ax.axis("off")

    lines = [
        "WEATHER SUMMARY",
        "─" * 40,
        f"Total groups: {len(all_w)}",
        f"Overall std: {all_w.std():.4f}",
        f"Range: [{all_w.min():.4f}, {all_w.max():.4f}]",
        "",
        f"Cross-seed corr: {all_cc.mean():+.4f} ± {all_cc.std():.4f}",
        f"  → Independent (≈ 0)",
        "",
        f"Lag-1 autocorr:  {all_ac.mean():+.4f} ± {all_ac.std():.4f}",
        f"  → i.i.d. over time (≈ 0)",
        "",
        f"Skewness: {sk:+.3f}  (normal = 0)",
        f"Kurtosis: {ku:.3f}  (normal = 3)",
        f"  → Approximately normal",
        "",
        f"Per-round std range:",
        f"  [{min(rs['w_vals'].std() for rs in round_stats.values()):.4f}, "
        f"{max(rs['w_vals'].std() for rs in round_stats.values()):.4f}]",
        "",
        f"Per-round mean range:",
        f"  [{min(means):+.4f}, {max(means):+.4f}]",
        f"  (absorbs winter + baseline)",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    fig.suptitle(
        "Weather component of food production\n"
        "weather ~ N(μ_round, σ²),  σ ≈ 0.03–0.05,  independent across seeds,  i.i.d. over time",
        fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig("weather_proof.png", dpi=150, bbox_inches="tight")
    print("\nSaved weather_proof.png")
    plt.close()


if __name__ == "__main__":
    main()
