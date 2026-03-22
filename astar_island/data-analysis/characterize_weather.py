"""
Characterize the weather component of the food formula.

Model: food_delta = pl*n_pl*(1-f) + fo*n_fo*(1-f) + fb*f*(1-f) - pop_rate*pop + weather
Weather is constant within a (seed, step) group but varies across groups.

Strategy:
1. Fit the model per round with outlier removal
2. Estimate weather per (seed, step) group as the group mean residual
3. Analyze weather distribution: by round, by step, by seed
4. Is weather i.i.d. or does it have temporal structure (autocorrelation)?
5. What is the range, variance, distribution shape?

Reproduce:  python3 data-analysis/characterize_weather.py
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


def estimate_weather(arr, groups, coeffs):
    """Given fitted coefficients, estimate weather per (seed, step) group."""
    pop, food = arr[:, 0], arr[:, 1]
    npl, nfo = arr[:, 2], arr[:, 3]
    fd = arr[:, 4]
    omf = 1 - food

    pred_no_weather = coeffs[0] * npl * omf + coeffs[1] * nfo * omf + coeffs[2] * food * omf + coeffs[3] * pop
    residual = fd - pred_no_weather

    # Weather = group mean of residual
    gmap = defaultdict(list)
    for i, g in enumerate(groups):
        gmap[g].append(i)

    weather = {}
    for g, indices in gmap.items():
        weather[g] = np.mean(residual[indices])

    return weather


def iterative_fit(arr, groups, n_iters=3, threshold=3.0):
    """Fit base model with iterative outlier removal."""
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


def main():
    rounds = load_replays_by_round()

    n_rounds = len(rounds)
    fig1, axes1 = plt.subplots(3, 3, figsize=(20, 16))
    axes1 = axes1.flatten()

    fig2, axes2 = plt.subplots(3, 3, figsize=(20, 16))
    axes2 = axes2.flatten()

    all_weather_by_round = {}

    print("=" * 80)
    print("WEATHER CHARACTERIZATION PER ROUND")
    print("  weather = group mean of (food_delta - predicted)")
    print("=" * 80)

    for idx, rid in enumerate(sorted(rounds)):
        arr, groups = collect_stable(rounds[rid])
        coeffs, r2, mask = iterative_fit(arr, groups)

        arr_clean = arr[mask]
        groups_clean = [groups[i] for i in range(len(groups)) if mask[i]]

        weather = estimate_weather(arr_clean, groups_clean, coeffs)
        all_weather_by_round[rid] = weather

        # Organize by seed and step
        seeds = sorted(set(s for s, st in weather.keys()))
        steps = sorted(set(st for s, st in weather.keys()))

        w_vals = np.array(list(weather.values()))

        print(f"\n  {rid[:12]} (R²={r2:.4f}, {len(weather)} groups)")
        print(f"    mean={w_vals.mean():.6f}  std={w_vals.std():.6f}")
        print(f"    min={w_vals.min():.6f}  max={w_vals.max():.6f}")
        print(f"    P05={np.percentile(w_vals, 5):.6f}  P95={np.percentile(w_vals, 95):.6f}")

        # Check quantization
        w_1000 = w_vals * 1000
        w_round = np.round(w_1000)
        quant_err = np.abs(w_1000 - w_round)
        pct_exact = np.mean(quant_err < 0.05) * 100
        print(f"    quantized to 0.001? {pct_exact:.1f}% within ±0.00005")

        # Check if weather is the same across seeds within a round
        # (i.e., does the same step have the same weather regardless of seed?)
        step_weather = defaultdict(list)
        for (seed, step), w in weather.items():
            step_weather[step].append(w)

        # For each step, check variance across seeds
        cross_seed_var = []
        step_means = {}
        for step in sorted(step_weather):
            vals = step_weather[step]
            if len(vals) > 1:
                cross_seed_var.append(np.var(vals))
                step_means[step] = np.mean(vals)

        if cross_seed_var:
            avg_cross_seed_std = np.sqrt(np.mean(cross_seed_var))
            print(f"    cross-seed std (same step): {avg_cross_seed_std:.6f}")
            if avg_cross_seed_std < 0.001:
                print(f"    → Weather is THE SAME across seeds!")
            else:
                print(f"    → Weather VARIES across seeds")

        # Check if weather is the same across seeds (pairwise correlation)
        if len(seeds) >= 2:
            seed_series = {}
            for seed in seeds:
                series = {}
                for (s, st), w in weather.items():
                    if s == seed:
                        series[st] = w
                seed_series[seed] = series

            # Pairwise correlation between seed weather time series
            corrs = []
            for i, s1 in enumerate(seeds):
                for s2 in seeds[i+1:]:
                    common_steps = sorted(set(seed_series[s1].keys()) & set(seed_series[s2].keys()))
                    if len(common_steps) > 5:
                        v1 = [seed_series[s1][st] for st in common_steps]
                        v2 = [seed_series[s2][st] for st in common_steps]
                        corr = np.corrcoef(v1, v2)[0, 1]
                        corrs.append(corr)
            if corrs:
                print(f"    cross-seed correlation: mean={np.mean(corrs):.4f}  "
                      f"range=[{np.min(corrs):.4f}, {np.max(corrs):.4f}]")

        # Plot 1: Weather time series per seed
        ax1 = axes1[idx]
        for seed in seeds:
            steps_s = sorted([st for (s, st) in weather if s == seed])
            w_s = [weather[(seed, st)] for st in steps_s]
            ax1.plot(steps_s, w_s, ".-", markersize=3, lw=0.8, alpha=0.7, label=f"seed {seed}")
        ax1.axhline(0, color="gray", lw=0.5)
        ax1.set_xlabel("step")
        ax1.set_ylabel("weather")
        ax1.set_title(f"{rid[:8]}  std={w_vals.std():.4f}")
        ax1.set_ylim(-0.15, 0.15)
        if len(seeds) <= 6:
            ax1.legend(fontsize=6, ncol=2)

        # Plot 2: Weather histogram
        ax2 = axes2[idx]
        ax2.hist(w_vals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax2.axvline(0, color="red", lw=1)
        ax2.axvline(w_vals.mean(), color="orange", lw=1, ls="--", label=f"mean={w_vals.mean():.4f}")
        ax2.set_xlabel("weather")
        ax2.set_ylabel("count")
        ax2.set_title(f"{rid[:8]}  std={w_vals.std():.4f}")
        ax2.legend(fontsize=7)

    fig1.suptitle("Weather time series per seed (each line = one seed)", fontsize=14)
    plt.tight_layout()
    fig1.savefig("weather_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("\nSaved weather_timeseries.png")

    fig2.suptitle("Weather distribution per round", fontsize=14)
    plt.tight_layout()
    fig2.savefig("weather_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved weather_histogram.png")

    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("AGGREGATE WEATHER ANALYSIS")
    print(f"{'='*80}")

    # Pooled weather values
    all_w = []
    all_steps = []
    all_seeds = []
    all_rids = []
    for rid, weather in all_weather_by_round.items():
        for (seed, step), w in weather.items():
            all_w.append(w)
            all_steps.append(step)
            all_seeds.append(seed)
            all_rids.append(rid)

    all_w = np.array(all_w)
    all_steps = np.array(all_steps)

    print(f"\n  Total weather estimates: {len(all_w)}")
    print(f"  Overall: mean={all_w.mean():.6f}  std={all_w.std():.6f}")
    print(f"  Skewness: {float(np.mean(((all_w - all_w.mean()) / all_w.std()) ** 3)):.4f}")
    print(f"  Kurtosis: {float(np.mean(((all_w - all_w.mean()) / all_w.std()) ** 4)):.4f} (normal=3)")

    # Weather vs step (pooled)
    print(f"\n  Weather by step (pooled across rounds and seeds):")
    step_means_all = []
    step_stds_all = []
    unique_steps = sorted(set(all_steps))
    for st in unique_steps:
        mask = all_steps == st
        if mask.sum() >= 3:
            step_means_all.append(all_w[mask].mean())
            step_stds_all.append(all_w[mask].std())

    step_means_all = np.array(step_means_all)
    step_stds_all = np.array(step_stds_all)
    print(f"    mean(step_means) = {step_means_all.mean():.6f}")
    print(f"    std(step_means) = {step_means_all.std():.6f}")
    print(f"    mean(step_stds) = {step_stds_all.mean():.6f}")
    if step_means_all.std() > 0.001:
        print(f"    → Weather mean VARIES by step")
    else:
        print(f"    → Weather mean is CONSTANT across steps")

    # Autocorrelation
    print(f"\n  Autocorrelation analysis:")
    acorrs_by_seed = []
    for rid, weather in all_weather_by_round.items():
        seeds_in_round = sorted(set(s for s, st in weather.keys()))
        for seed in seeds_in_round:
            steps_s = sorted([st for (s, st) in weather if s == seed])
            if len(steps_s) < 10:
                continue
            w_s = np.array([weather[(seed, st)] for st in steps_s])
            w_s_centered = w_s - w_s.mean()
            var = np.var(w_s_centered)
            if var < 1e-15:
                continue
            # Lag-1 autocorrelation
            ac1 = np.corrcoef(w_s_centered[:-1], w_s_centered[1:])[0, 1]
            # Lag-2
            ac2 = np.corrcoef(w_s_centered[:-2], w_s_centered[2:])[0, 1] if len(w_s) > 4 else np.nan
            acorrs_by_seed.append((ac1, ac2, rid[:8], seed))

    if acorrs_by_seed:
        ac1s = [a[0] for a in acorrs_by_seed]
        ac2s = [a[1] for a in acorrs_by_seed if not np.isnan(a[1])]
        print(f"    Lag-1: mean={np.mean(ac1s):.4f}  range=[{np.min(ac1s):.4f}, {np.max(ac1s):.4f}]")
        print(f"    Lag-2: mean={np.mean(ac2s):.4f}  range=[{np.min(ac2s):.4f}, {np.max(ac2s):.4f}]")
        if abs(np.mean(ac1s)) > 0.1:
            print(f"    → Weather has temporal correlation (not i.i.d.)")
        else:
            print(f"    → Weather is approximately i.i.d.")

    # ═══════════════════════════════════════════════════════════════════
    # DEEP DIVE: Is weather truly shared across seeds at the same step?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DEEP DIVE: Is weather shared across seeds?")
    print(f"{'='*80}")

    fig3, axes3 = plt.subplots(3, 3, figsize=(20, 16))
    axes3 = axes3.flatten()

    for idx, rid in enumerate(sorted(all_weather_by_round)):
        weather = all_weather_by_round[rid]
        seeds_in_round = sorted(set(s for s, st in weather.keys()))

        ax = axes3[idx]

        if len(seeds_in_round) >= 2:
            # Plot seed 0 vs seed 1 weather at same step
            s0, s1 = seeds_in_round[0], seeds_in_round[1]
            common = sorted(set(st for s, st in weather if s == s0) &
                          set(st for s, st in weather if s == s1))
            if common:
                w0 = [weather[(s0, st)] for st in common]
                w1 = [weather[(s1, st)] for st in common]
                corr = np.corrcoef(w0, w1)[0, 1]
                ax.scatter(w0, w1, s=8, alpha=0.6, color="steelblue")
                lim = max(abs(min(min(w0), min(w1))), abs(max(max(w0), max(w1))))
                lim = min(lim, 0.15)
                ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xlabel(f"seed {s0}")
                ax.set_ylabel(f"seed {s1}")
                ax.set_title(f"{rid[:8]}  corr={corr:.3f}")
                ax.set_aspect("equal")

    fig3.suptitle("Weather: seed 0 vs seed 1 at same step\n(if shared, points lie on y=x)", fontsize=14)
    plt.tight_layout()
    fig3.savefig("weather_cross_seed.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved weather_cross_seed.png")

    # ═══════════════════════════════════════════════════════════════════
    # Check if weather depends on step (seasonal pattern?)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("SEASONAL PATTERN: Weather vs step")
    print(f"{'='*80}")

    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 5))

    for rid in sorted(all_weather_by_round):
        weather = all_weather_by_round[rid]
        step_w = defaultdict(list)
        for (seed, step), w in weather.items():
            step_w[step].append(w)
        steps = sorted(step_w)
        means = [np.mean(step_w[st]) for st in steps]
        ax4.plot(steps, means, ".-", alpha=0.6, markersize=3, label=rid[:8])

    ax4.axhline(0, color="gray", lw=0.5)
    ax4.set_xlabel("step")
    ax4.set_ylabel("mean weather")
    ax4.set_title("Mean weather by step (one line per round)")
    ax4.legend(fontsize=7, ncol=3)
    ax4.set_ylim(-0.15, 0.15)
    plt.tight_layout()
    fig4.savefig("weather_vs_step.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("Saved weather_vs_step.png")

    # Print step-by-step for a few rounds
    for rid in sorted(all_weather_by_round)[:2]:
        weather = all_weather_by_round[rid]
        seeds_in = sorted(set(s for s, st in weather.keys()))
        print(f"\n  {rid[:12]} — weather values (step × seed):")
        print(f"    {'step':>4s}", end="")
        for seed in seeds_in:
            print(f"  seed_{seed:>2d}", end="")
        print(f"  {'mean':>8s}  {'std':>7s}")

        steps_in = sorted(set(st for s, st in weather.keys()))
        for step in steps_in[:25]:  # first 25 steps
            print(f"    {step:>4d}", end="")
            vals = []
            for seed in seeds_in:
                w = weather.get((seed, step))
                if w is not None:
                    print(f"  {w:>8.4f}", end="")
                    vals.append(w)
                else:
                    print(f"  {'—':>8s}", end="")
            if vals:
                print(f"  {np.mean(vals):>8.4f}  {np.std(vals):>7.4f}")
            else:
                print()


if __name__ == "__main__":
    main()
