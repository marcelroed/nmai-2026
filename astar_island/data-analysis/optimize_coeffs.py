"""
Optimize food formula coefficients using scipy.optimize.

Model: food_delta = pl*n_pl*(1-f) + fo*n_fo*(1-f) + fb*f*(1-f) - pop_rate*pop + weather_{seed,step}

Strategy:
1. For given (pl, fo, fb, pop_rate), weather is estimated per group as group mean residual
2. Minimize total MAE (or SSE) across all clean observations

Reproduce:  python3 data-analysis/optimize_coeffs.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

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


def build_group_indices(groups):
    gmap = defaultdict(list)
    for i, g in enumerate(groups):
        gmap[g].append(i)
    return dict(gmap)


def compute_sse(params, arr, group_indices):
    pl, fo, fb, pop_rate = params
    pop = arr[:, 0]
    food = arr[:, 1]
    n_pl, n_fo = arr[:, 2], arr[:, 3]
    fd = arr[:, 7]
    omf = 1 - food

    pred = pl * n_pl * omf + fo * n_fo * omf + fb * food * omf - pop_rate * pop
    residual = fd - pred

    # Estimate weather per group (group mean of residual)
    for indices in group_indices.values():
        weather = residual[indices].mean()
        residual[indices] -= weather

    return np.sum(residual ** 2)


def compute_mae(params, arr, group_indices):
    pl, fo, fb, pop_rate = params
    pop = arr[:, 0]
    food = arr[:, 1]
    n_pl, n_fo = arr[:, 2], arr[:, 3]
    fd = arr[:, 7]
    omf = 1 - food

    pred = pl * n_pl * omf + fo * n_fo * omf + fb * food * omf - pop_rate * pop
    residual = fd - pred

    for indices in group_indices.values():
        weather = residual[indices].mean()
        residual[indices] -= weather

    return np.mean(np.abs(residual))


def iterative_optimize(arr, groups, n_iters=3, threshold=3.0):
    """Optimize with iterative outlier removal."""
    group_indices = build_group_indices(groups)

    for iteration in range(n_iters + 1):
        # Optimize
        x0 = [0.043, 0.075, 0.9, 0.113]
        result = minimize(compute_sse, x0, args=(arr, group_indices),
                         method='Nelder-Mead',
                         options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 50000})
        params = result.x

        if iteration < n_iters:
            # Compute residuals and remove outliers
            pl, fo, fb, pop_rate = params
            pop = arr[:, 0]
            food = arr[:, 1]
            n_pl, n_fo = arr[:, 2], arr[:, 3]
            fd = arr[:, 7]
            omf = 1 - food
            pred = pl * n_pl * omf + fo * n_fo * omf + fb * food * omf - pop_rate * pop
            residual = fd - pred
            for indices in group_indices.values():
                weather = residual[indices].mean()
                residual[indices] -= weather
            mae = np.mean(np.abs(residual))
            inlier = np.abs(residual) < threshold * mae
            n_removed = (~inlier).sum()
            arr = arr[inlier]
            groups = [groups[i] for i in range(len(groups)) if inlier[i]]
            group_indices = build_group_indices(groups)

    return params, arr, groups


def main():
    rounds = load_replays_by_round()

    print("=" * 80)
    print("OPTIMIZED COEFFICIENTS PER ROUND")
    print("  food_delta = pl*n_pl*(1-f) + fo*n_fo*(1-f) + fb*f*(1-f) - pop*pop + weather")
    print("=" * 80)

    all_params = []

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        n_orig = len(arr)

        params, arr_clean, groups_clean = iterative_optimize(arr, groups)
        pl, fo, fb, pop_rate = params
        n_clean = len(arr_clean)

        # Compute final MAE and R²
        group_indices = build_group_indices(groups_clean)
        mae = compute_mae(params, arr_clean, group_indices)

        # R²
        pop = arr_clean[:, 0]
        food = arr_clean[:, 1]
        n_pl, n_fo = arr_clean[:, 2], arr_clean[:, 3]
        fd = arr_clean[:, 7]
        omf = 1 - food
        pred = pl * n_pl * omf + fo * n_fo * omf + fb * food * omf - pop_rate * pop
        residual = fd - pred
        for indices in group_indices.values():
            weather = residual[indices].mean()
            residual[indices] -= weather
        ss_res = np.sum(residual ** 2)

        # Demean fd for ss_tot
        fd_dm = fd.copy()
        for indices in group_indices.values():
            fd_dm[indices] -= fd[indices].mean()
        ss_tot = np.sum(fd_dm ** 2)
        r2 = 1 - ss_res / ss_tot

        ratio = fo / pl
        print(f"\n  {rid[:12]} ({n_orig} → {n_clean} obs)")
        print(f"    pl={pl:.6f} fo={fo:.6f} fb={fb:.6f} pop={pop_rate:.6f}")
        print(f"    fo/pl={ratio:.4f}")
        print(f"    R²={r2:.6f} MAE={mae:.6f}")

        # Check exact match rate
        exact = np.mean(np.abs(residual) < 0.0005)
        close = np.mean(np.abs(residual) < 0.0015)
        print(f"    exact (±0.0005): {100*exact:.1f}%  close (±0.0015): {100*close:.1f}%")

        all_params.append(params)

    all_params = np.array(all_params)

    print(f"\n\n{'='*80}")
    print("SUMMARY ACROSS ROUNDS")
    print(f"{'='*80}")
    names = ["pl", "fo", "fb", "pop"]
    for i, name in enumerate(names):
        vals = all_params[:, i]
        print(f"  {name:>5s}: mean={vals.mean():.6f} ± {vals.std():.6f} "
              f"(CV={vals.std()/abs(vals.mean())*100:.1f}%) "
              f"range=[{vals.min():.6f}, {vals.max():.6f}]")

    ratios = all_params[:, 1] / all_params[:, 0]
    print(f"\n  fo/pl: mean={ratios.mean():.4f} ± {ratios.std():.4f}")
    print(f"         per-round: {', '.join(f'{r:.3f}' for r in ratios)}")

    # Test clean fraction candidates
    print(f"\n\n{'='*80}")
    print("CLEAN NUMBER CANDIDATES")
    print(f"{'='*80}")
    m = all_params.mean(axis=0)
    print(f"  Fitted means: pl={m[0]:.6f}  fo={m[1]:.6f}  fb={m[2]:.6f}  pop={m[3]:.6f}")

    candidates = [
        ("pl=1/24,fo=3/40", 1/24, 3/40),
        ("pl=0.04,fo=0.07", 0.04, 0.07),
        ("pl=0.04,fo=0.075", 0.04, 0.075),
        ("pl=0.043,fo=0.075", 0.043, 0.075),
        ("pl=0.045,fo=0.08", 0.045, 0.08),
        ("pl=0.04,fo=0.08", 0.04, 0.08),
        ("pl=1/25,fo=3/40", 1/25, 3/40),
    ]

    for label, pl_cand, fo_cand in candidates:
        total_sse = 0
        total_n = 0
        for rid in sorted(rounds):
            replays = rounds[rid]
            arr, groups = collect_stable_rows(replays)
            # Use optimizer for fb and pop with fixed pl, fo
            group_indices = build_group_indices(groups)
            def sse_fixed(params2):
                return compute_sse([pl_cand, fo_cand, params2[0], params2[1]],
                                  arr, group_indices)
            res = minimize(sse_fixed, [0.9, 0.113], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'maxiter': 10000})
            total_sse += res.fun
            total_n += len(arr)
        rmse = np.sqrt(total_sse / total_n)
        print(f"  {label:<25s}: RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
