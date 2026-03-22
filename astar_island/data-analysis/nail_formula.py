"""
Nail down the exact food formula.

From previous analysis:
  food_delta = pl_rate*n_pl*(1-f) + fo_rate*n_fo*(1-f) + feedback*f*(1-f) - pop_rate*pop + weather

Questions remaining:
1. Do mountain and ocean contribute?
2. Do adjacent settlements contribute?
3. Is the fo/pl ratio exactly 1.5 or 2.0?
4. Are coefficients truly constant across rounds?
5. What are the exact values (are they clean numbers)?

Reproduce:  python3 data-analysis/nail_formula.py
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


def iterative_clean(arr, groups, build_features_fn, n_iters=3, threshold=3.0):
    """Iteratively remove outliers > threshold*MAE."""
    for _ in range(n_iters):
        X, y = build_features_fn(arr)
        dm_X = np.zeros_like(X)
        dm_y = demean_col(y, groups)
        for col in range(X.shape[1]):
            dm_X[:, col] = demean_col(X[:, col], groups)
        c, r2, mae, res = fit_ols(dm_X, dm_y)
        inlier = np.abs(res) < threshold * mae
        arr = arr[inlier]
        groups = [groups[i] for i in range(len(groups)) if inlier[i]]
    # Final fit
    X, y = build_features_fn(arr)
    dm_X = np.zeros_like(X)
    dm_y = demean_col(y, groups)
    for col in range(X.shape[1]):
        dm_X[:, col] = demean_col(X[:, col], groups)
    c, r2, mae, res = fit_ols(dm_X, dm_y)
    return c, r2, mae, res, arr, groups


def main():
    rounds = load_replays_by_round()

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Full model with mt, oc, st
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PART 1: Full terrain model (pl, fo, mt, oc, st)")
    print("=" * 80)

    def features_full(arr):
        pop, food = arr[:, 0], arr[:, 1]
        npl, nfo, nmt, noc, nst = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]
        omf = 1 - food
        X = np.column_stack([npl*omf, nfo*omf, nmt*omf, noc*omf, nst*omf, food*omf, pop])
        y = arr[:, 7]
        return X, y

    def features_base(arr):
        pop, food = arr[:, 0], arr[:, 1]
        npl, nfo = arr[:, 2], arr[:, 3]
        omf = 1 - food
        X = np.column_stack([npl*omf, nfo*omf, food*omf, pop])
        y = arr[:, 7]
        return X, y

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)

        c_full, r2_full, mae_full, _, arr_c, grp_c = iterative_clean(
            arr.copy(), list(groups), features_full)
        c_base, r2_base, mae_base, _, _, _ = iterative_clean(
            arr.copy(), list(groups), features_base)

        print(f"\nRound {rid[:12]} ({len(arr)} → {len(arr_c)} obs)")
        print(f"  Full: R²={r2_full:.6f} MAE={mae_full:.6f}")
        print(f"    pl={c_full[0]:.6f} fo={c_full[1]:.6f} mt={c_full[2]:.6f} "
              f"oc={c_full[3]:.6f} st={c_full[4]:.6f}")
        print(f"    f*(1-f)={c_full[5]:.6f} pop={c_full[6]:.6f}")
        print(f"  Base: R²={r2_base:.6f} MAE={mae_base:.6f}")
        print(f"  ΔR²={r2_full - r2_base:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Test specific fo/pl ratios
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 2: Test specific fo/pl ratios")
    print(f"{'='*80}")

    for alpha in [1.0, 1.5, 1.75, 2.0]:
        print(f"\n  Testing α = {alpha}:")

        def make_features(alpha_val):
            def features_prop(arr):
                pop, food = arr[:, 0], arr[:, 1]
                npl, nfo = arr[:, 2], arr[:, 3]
                omf = 1 - food
                terrain_score = npl + alpha_val * nfo
                X = np.column_stack([terrain_score * omf, food * omf, pop])
                y = arr[:, 7]
                return X, y
            return features_prop

        for rid in sorted(rounds):
            replays = rounds[rid]
            arr, groups = collect_stable_rows(replays)
            c_prop, r2_prop, _, _, _, _ = iterative_clean(
                arr.copy(), list(groups), make_features(alpha))
            c_free, r2_free, _, _, _, _ = iterative_clean(
                arr.copy(), list(groups), features_base)
            print(f"    {rid[:12]}: prop R²={r2_prop:.6f} free R²={r2_free:.6f} "
                  f"Δ={r2_free-r2_prop:.6f} terrain_rate={c_prop[0]:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Test if coefficients can be fixed across rounds
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 3: Fixed vs free coefficients across rounds")
    print(f"{'='*80}")

    # Collect all clean data
    all_arr = []
    all_groups = []
    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        # Tag groups to separate rounds
        tagged = [(rid, g[0], g[1]) for g in groups]
        all_arr.append(arr)
        all_groups.extend(tagged)

    pooled = np.vstack(all_arr)

    def features_pooled(arr):
        pop, food = arr[:, 0], arr[:, 1]
        npl, nfo = arr[:, 2], arr[:, 3]
        omf = 1 - food
        X = np.column_stack([npl*omf, nfo*omf, food*omf, pop])
        y = arr[:, 7]
        return X, y

    c_pool, r2_pool, mae_pool, _, pool_clean, pool_grp = iterative_clean(
        pooled.copy(), list(all_groups), features_pooled)

    print(f"\nPooled model (all rounds, {len(pooled)} → {len(pool_clean)} obs)")
    print(f"  R²={r2_pool:.6f} MAE={mae_pool:.6f}")
    print(f"  pl*(1-f)={c_pool[0]:.6f} fo*(1-f)={c_pool[1]:.6f}")
    print(f"  f*(1-f)={c_pool[2]:.6f} pop={c_pool[3]:.6f}")
    print(f"  fo/pl ratio={c_pool[1]/c_pool[0]:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: Test "nice" coefficient values
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 4: Testing 'nice' coefficient values")
    print("  Does the formula use clean fractions?")
    print(f"{'='*80}")

    # Test candidates for each coefficient
    pl_candidates = [0.04, 0.042, 0.043, 0.045, 1/24, 1/25, 0.05]
    fo_candidates = [0.07, 0.074, 0.075, 0.08, 3/40, 1/13]
    fb_candidates = [0.9, 0.95, 1.0, 0.914]
    pop_candidates = [-0.10, -0.11, -0.113, -0.115, -0.12, -1/9, -1/8]

    print(f"\n  Reference values: pl={c_pool[0]:.6f} fo={c_pool[1]:.6f} "
          f"fb={c_pool[2]:.6f} pop={c_pool[3]:.6f}")

    # For each round, compute residual using candidate coefficients
    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)

        # Clean data first
        _, _, _, _, arr_c, grp_c = iterative_clean(
            arr.copy(), list(groups), features_base)

        pop = arr_c[:, 0]
        food = arr_c[:, 1]
        npl, nfo = arr_c[:, 2], arr_c[:, 3]
        fd = arr_c[:, 7]
        omf = 1 - food

        # Compute food_delta - terrain_effect, then demean
        # If we fix terrain coefficients, we need to fit the remaining (f*(1-f), pop)
        # from the residual

        for pl_val in [c_pool[0]]:
            for fo_val in [c_pool[1]]:
                terrain_contrib = pl_val * npl * omf + fo_val * nfo * omf
                adjusted = fd - terrain_contrib

                # Fit f*(1-f) and pop from adjusted
                feat = np.column_stack([food * omf, pop, adjusted])
                dm = np.zeros_like(feat)
                for col in range(feat.shape[1]):
                    dm[:, col] = demean_col(feat[:, col], grp_c)
                c2, r2_2, mae_2, _ = fit_ols(dm[:, :2], dm[:, 2])
                # This gives us the "free" f*(1-f) and pop for this round
                # with fixed terrain

    # Instead, let's test rounding
    print("\n  Testing if coefficients are rounded to specific precision:")

    # Check if food values suggest a specific quantization
    for rid in sorted(rounds)[:1]:
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        _, _, _, _, arr_c, grp_c = iterative_clean(
            arr.copy(), list(groups), features_base)
        food = arr_c[:, 1]
        fd = arr_c[:, 7]
        # Check food precision
        food_frac = food * 1000
        food_int = np.round(food_frac)
        food_error = np.max(np.abs(food_frac - food_int))
        print(f"  Food quantization: max error from integer×0.001 = {food_error:.6f}")

        fd_frac = fd * 1000
        fd_int = np.round(fd_frac)
        fd_error = np.max(np.abs(fd_frac - fd_int))
        print(f"  Food_delta quantization: max error from integer×0.001 = {fd_error:.6f}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 5: Fit the model with constrained terrain, free per-round feedback/pop
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 5: Fixed terrain + free per-round feedback/pop")
    print(f"{'='*80}")

    pl_fixed = c_pool[0]
    fo_fixed = c_pool[1]

    print(f"  Fixed: pl*(1-f)={pl_fixed:.6f}, fo*(1-f)={fo_fixed:.6f}")
    print(f"\n  {'Round':<15} {'f*(1-f)':>10} {'pop':>10} {'R²':>10} {'MAE':>10}")

    for rid in sorted(rounds):
        replays = rounds[rid]
        arr, groups = collect_stable_rows(replays)
        _, _, _, _, arr_c, grp_c = iterative_clean(
            arr.copy(), list(groups), features_base)

        pop = arr_c[:, 0]
        food = arr_c[:, 1]
        npl, nfo = arr_c[:, 2], arr_c[:, 3]
        fd = arr_c[:, 7]
        omf = 1 - food

        terrain_contrib = pl_fixed * npl * omf + fo_fixed * nfo * omf
        adjusted = fd - terrain_contrib

        feat = np.column_stack([food * omf, pop, adjusted])
        dm = np.zeros_like(feat)
        for col in range(feat.shape[1]):
            dm[:, col] = demean_col(feat[:, col], grp_c)
        c_adj, r2_adj, mae_adj, _ = fit_ols(dm[:, :2], dm[:, 2])

        # Compare with free terrain
        feat_free = np.column_stack([npl*omf, nfo*omf, food*omf, pop, fd])
        dm_free = np.zeros_like(feat_free)
        for col in range(feat_free.shape[1]):
            dm_free[:, col] = demean_col(feat_free[:, col], grp_c)
        _, r2_free, _, _ = fit_ols(dm_free[:, :4], dm_free[:, 4])

        print(f"  {rid[:14]:<15} {c_adj[0]:>10.6f} {c_adj[1]:>10.6f} "
              f"{r2_adj:>10.6f} {mae_adj:>10.6f}  (free: {r2_free:.6f}, Δ={r2_free-r2_adj:.6f})")

    # ═══════════════════════════════════════════════════════════════════
    # PART 6: Step-0 analysis to isolate terrain rate precisely
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("PART 6: Step 0→1 terrain rate (food~=0.5, pop varies)")
    print("  At step 0, all settlements start fresh. Within a seed, weather is constant.")
    print("  So food_delta should be a PURE function of terrain + food + pop + constant.")
    print(f"{'='*80}")

    for rid in sorted(rounds):
        replays = rounds[rid]
        for replay in replays[:2]:  # first 2 seeds
            fb, fa = replay["frames"][0], replay["frames"][1]
            grid = fb["grid"]
            seed = replay["seed_index"]

            foods_b, foods_d, pops_b = [], [], []
            npls, nfos, nmts, nocs, nsts = [], [], [], [], []

            before_map = {(s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]}
            after_map = {(s["x"], s["y"]): s for s in fa["settlements"] if s["alive"]}

            for pos, sb in before_map.items():
                sa = after_map.get(pos)
                if not sa or sa["owner_id"] != sb["owner_id"]:
                    continue
                x, y = pos
                if grid[y][x] == PORT:
                    continue
                adj = terrain_adj_full(grid, x, y)
                foods_b.append(sb["food"])
                foods_d.append(sa["food"] - sb["food"])
                pops_b.append(sb["population"])
                npls.append(adj.get(PLAINS, 0))
                nfos.append(adj.get(FOREST, 0))
                nmts.append(adj.get(MOUNTAIN, 0))
                nocs.append(adj.get(OCEAN, 0))
                nsts.append(adj.get(SETTLEMENT, 0) + adj.get(PORT, 0) + adj.get(RUIN, 0))

            fb_arr = np.array(foods_b)
            fd_arr = np.array(foods_d)
            pop_arr = np.array(pops_b)
            pl_arr = np.array(npls, dtype=float)
            fo_arr = np.array(nfos, dtype=float)
            mt_arr = np.array(nmts, dtype=float)
            oc_arr = np.array(nocs, dtype=float)
            st_arr = np.array(nsts, dtype=float)
            omf_arr = 1 - fb_arr

            # Full model within this single seed (no demeaning needed, just intercept)
            X = np.column_stack([
                np.ones(len(fd_arr)),
                pl_arr * omf_arr, fo_arr * omf_arr,
                mt_arr * omf_arr, oc_arr * omf_arr, st_arr * omf_arr,
                fb_arr * omf_arr,
                pop_arr,
            ])
            c, r2, mae, res = fit_ols(X, fd_arr)

            if seed == 0:
                print(f"\n  Round {rid[:12]} seed={seed}: {len(fd_arr)} settlements")
                print(f"    food: mean={fb_arr.mean():.4f} std={fb_arr.std():.4f} "
                      f"range=[{fb_arr.min():.4f}, {fb_arr.max():.4f}]")
                print(f"    pop:  mean={pop_arr.mean():.4f} std={pop_arr.std():.4f} "
                      f"range=[{pop_arr.min():.4f}, {pop_arr.max():.4f}]")
                print(f"    R²={r2:.6f} MAE={mae:.6f}")
                print(f"    intercept={c[0]:.6f}")
                print(f"    pl*(1-f)={c[1]:.6f} fo*(1-f)={c[2]:.6f} mt*(1-f)={c[3]:.6f} "
                      f"oc*(1-f)={c[4]:.6f} st*(1-f)={c[5]:.6f}")
                print(f"    f*(1-f)={c[6]:.6f} pop={c[7]:.6f}")

                if abs(c[1]) > 1e-6:
                    print(f"    fo/pl ratio={c[2]/c[1]:.4f}")


if __name__ == "__main__":
    main()
