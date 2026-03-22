"""
Grid search for exact food formula coefficients.

Since food is quantized to 0.001, we can test candidate coefficient values
and measure how well they predict food_delta exactly.

Strategy:
1. For each candidate (pl_rate, fo_rate, feedback, pop_rate):
2. For each (seed, step) group, estimate weather as the group-mean residual
3. Compute predicted food_delta for each observation
4. Compare with observed food_delta

Also: test whether the formula might be the linear form from the simulator:
  food_delta = food_base + pop_coeff*pop + feedback*food + terrain_terms

Reproduce:  python3 data-analysis/grid_search_coeffs.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def collect_stable_rows(replays):
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
                if sa["defense"] - sb["defense"] < -0.001:
                    continue
                if sa["food"] >= 0.997 or sa["food"] <= 0.001:
                    continue
                if sa["population"] - sb["population"] < -0.05:
                    continue
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
                ))
                groups.append((seed, step))
    return np.array(rows, dtype=np.float64), groups


def test_model(arr, groups, pl, fo, fb, pop_rate, mt=0, oc=0, st=0, model_type="logistic"):
    """Test a specific coefficient combination.

    model_type="logistic": food_delta = pl*npl*(1-f) + fo*nfo*(1-f) + fb*f*(1-f) - pop*pop + weather
    model_type="linear":   food_delta = pl*npl + fo*nfo + fb*food - pop*pop + weather
    """
    pop_arr = arr[:, 0]
    food = arr[:, 1]
    n_pl, n_fo, n_mt, n_oc, n_st = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
    fd = arr[:, 7]
    omf = 1 - food

    if model_type == "logistic":
        pred_no_weather = (pl * n_pl * omf + fo * n_fo * omf + mt * n_mt * omf
                          + oc * n_oc * omf + st * n_st * omf
                          + fb * food * omf - pop_rate * pop_arr)
    elif model_type == "linear":
        pred_no_weather = (pl * n_pl + fo * n_fo + mt * n_mt + oc * n_oc + st * n_st
                          + fb * food - pop_rate * pop_arr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    residual = fd - pred_no_weather

    # Estimate weather per group
    group_map = defaultdict(list)
    for i, g in enumerate(groups):
        group_map[g].append(i)

    weather = np.zeros(len(arr))
    for indices in group_map.values():
        w = residual[indices].mean()
        weather[indices] = w

    final_residual = residual - weather
    mae = np.mean(np.abs(final_residual))
    exact_match = np.mean(np.abs(final_residual) < 0.0005)  # within ±0.5 of 0.001
    r2_var = 1 - np.var(final_residual) / np.var(fd - np.mean(fd)) if np.var(fd) > 0 else 0

    return mae, exact_match, r2_var


def main():
    rounds = load_replays_by_round()

    # Use a single round with good R² for grid search
    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        if len(arr) > 10000:
            break

    print(f"Using round {rid[:12]} ({len(arr)} terrain-stable obs)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Grid search over (pl, fo, fb, pop) for logistic model
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("PART 1: Grid search — logistic model")
    print(f"{'='*72}")

    best_mae = 1.0
    best_params = None

    pl_range = np.arange(0.030, 0.060, 0.001)
    fo_range = np.arange(0.060, 0.100, 0.001)
    fb_range = np.arange(0.5, 1.2, 0.05)
    pop_range = np.arange(0.08, 0.16, 0.005)

    # Coarse grid search
    print("  Coarse grid search...")
    results = []
    for pl in pl_range:
        for fo in fo_range:
            for fb_val in fb_range:
                for pop_val in pop_range:
                    mae, exact, r2 = test_model(arr, groups, pl, fo, fb_val, pop_val)
                    results.append((mae, exact, r2, pl, fo, fb_val, pop_val))
                    if mae < best_mae:
                        best_mae = mae
                        best_params = (pl, fo, fb_val, pop_val)

    results.sort(key=lambda x: x[0])
    print(f"\n  Top 10 parameter sets:")
    for mae, exact, r2, pl, fo, fb_val, pop_val in results[:10]:
        ratio = fo / pl if pl > 0 else float('inf')
        print(f"    pl={pl:.3f} fo={fo:.3f} fb={fb_val:.2f} pop={pop_val:.3f}  "
              f"MAE={mae:.6f} exact={100*exact:.1f}% R²={r2:.4f} fo/pl={ratio:.2f}")

    # Fine grid around best
    print(f"\n  Fine grid around best ({best_params})...")
    bp = best_params
    pl_fine = np.arange(bp[0]-0.005, bp[0]+0.006, 0.0005)
    fo_fine = np.arange(bp[1]-0.005, bp[1]+0.006, 0.0005)
    fb_fine = np.arange(bp[2]-0.1, bp[2]+0.11, 0.01)
    pop_fine = np.arange(bp[3]-0.01, bp[3]+0.011, 0.001)

    results_fine = []
    for pl in pl_fine:
        for fo in fo_fine:
            for fb_val in fb_fine:
                for pop_val in pop_fine:
                    mae, exact, r2 = test_model(arr, groups, pl, fo, fb_val, pop_val)
                    results_fine.append((mae, exact, r2, pl, fo, fb_val, pop_val))

    results_fine.sort(key=lambda x: x[0])
    print(f"\n  Top 10 fine-grid parameter sets:")
    for mae, exact, r2, pl, fo, fb_val, pop_val in results_fine[:10]:
        ratio = fo / pl if pl > 0 else float('inf')
        print(f"    pl={pl:.4f} fo={fo:.4f} fb={fb_val:.3f} pop={pop_val:.4f}  "
              f"MAE={mae:.6f} exact={100*exact:.1f}% R²={r2:.4f} fo/pl={ratio:.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Same grid search for LINEAR model
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("PART 2: Grid search — linear model")
    print(f"{'='*72}")

    results_lin = []
    for pl in np.arange(0.020, 0.060, 0.002):
        for fo in np.arange(0.030, 0.080, 0.002):
            for fb_val in np.arange(-1.0, 0.5, 0.05):
                for pop_val in np.arange(0.05, 0.20, 0.005):
                    mae, exact, r2 = test_model(arr, groups, pl, fo, fb_val, pop_val,
                                                model_type="linear")
                    results_lin.append((mae, exact, r2, pl, fo, fb_val, pop_val))

    results_lin.sort(key=lambda x: x[0])
    print(f"\n  Top 10 linear model:")
    for mae, exact, r2, pl, fo, fb_val, pop_val in results_lin[:10]:
        print(f"    pl={pl:.3f} fo={fo:.3f} fb={fb_val:.2f} pop={pop_val:.3f}  "
              f"MAE={mae:.6f} exact={100*exact:.1f}% R²={r2:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Cross-validate best params on other rounds
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("PART 3: Cross-validate best logistic params on all rounds")
    print(f"{'='*72}")

    best = results_fine[0]
    pl_best, fo_best, fb_best, pop_best = best[3], best[4], best[5], best[6]

    for rid2 in sorted(rounds):
        replays2 = rounds[rid2]
        arr2, groups2 = collect_stable_rows(replays2)
        mae, exact, r2 = test_model(arr2, groups2, pl_best, fo_best, fb_best, pop_best)
        print(f"  {rid2[:12]}: MAE={mae:.6f} exact={100*exact:.1f}% R²={r2:.4f} ({len(arr2)} obs)")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: Per-round optimal coefficients
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*72}")
    print("PART 4: Per-round fine grid search")
    print(f"{'='*72}")

    for rid2 in sorted(rounds):
        replays2 = rounds[rid2]
        arr2, groups2 = collect_stable_rows(replays2)

        best_r2 = -1
        best_p = None
        for pl in np.arange(0.030, 0.060, 0.002):
            for fo in np.arange(0.055, 0.100, 0.002):
                for fb_val in np.arange(0.5, 1.15, 0.05):
                    for pop_val in np.arange(0.080, 0.155, 0.005):
                        mae, exact, r2 = test_model(arr2, groups2, pl, fo, fb_val, pop_val)
                        if mae < (best_r2 if best_p else 1.0):
                            best_r2 = mae
                            best_p = (pl, fo, fb_val, pop_val, mae, exact, r2)

        if best_p:
            pl, fo, fb_val, pop_val, mae, exact, r2 = best_p
            ratio = fo / pl
            print(f"  {rid2[:12]}: pl={pl:.3f} fo={fo:.3f} fb={fb_val:.2f} pop={pop_val:.3f} "
                  f"MAE={mae:.6f} exact={100*exact:.1f}% fo/pl={ratio:.2f}")


if __name__ == "__main__":
    main()
