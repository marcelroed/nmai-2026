"""
Food analysis v2: restrict to low food_before to avoid phase-1 cap artifact,
and detect the cap value to determine winter_severity.

Reproduce:  python3 data-analysis/analyze_food_v2.py
"""

import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


def terrain_adj_counts(grid, x, y):
    h, w = len(grid), len(grid[0])
    np_, nf, nm, ns = 0, 0, 0, 0
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if nx == x and ny == y:
                continue
            t = grid[ny][nx]
            if t == PLAINS or t == EMPTY:
                np_ += 1
            elif t == FOREST:
                nf += 1
            elif t == MOUNTAIN:
                nm += 1
            elif t in (SETTLEMENT, PORT, RUIN):
                ns += 1
    return np_, nf, nm, ns


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
    """Return rows: (pop_b, food_b, wealth_b, food_a, n_pl, n_fo, n_mt, n_st, is_port, step)"""
    rows = []
    for replay in replays:
        frames = replay["frames"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            grid = fb["grid"]
            before_map = {
                (s["x"], s["y"]): s for s in fb["settlements"] if s["alive"]
            }
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
                adj = terrain_adj_counts(grid, x, y)
                is_port = 1 if grid[y][x] == PORT else 0

                rows.append((
                    sb["population"], sb["food"], sb["wealth"],
                    sa["food"],
                    *adj,
                    is_port, fa["step"],
                ))
    return rows


def fit_ols(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    res = y - pred
    mae = np.mean(np.abs(res))
    max_err = np.max(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, max_err, res


def main():
    rounds = load_replays_by_round()
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays from "
          f"{len(rounds)} rounds\n")

    for rid in sorted(rounds):
        replays = rounds[rid]
        rows = collect_rows(replays)
        arr = np.array(rows, dtype=np.float64)

        pop_b = arr[:, 0]
        food_b = arr[:, 1]
        food_a = arr[:, 3]
        food_delta = food_a - food_b
        is_port = arr[:, 8]

        # ── 1. Detect cap: find the most common food_after value for non-port ──
        non_port = is_port == 0
        food_a_np = food_a[non_port]
        # Round to 3 decimals to find clusters
        rounded = np.round(food_a_np, 3)
        vals, counts = np.unique(rounded, return_counts=True)
        top_idx = np.argsort(-counts)[:5]
        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}")
        print(f"{'=' * 72}")
        print(f"  Top 5 most common food_after values (non-port, rounded to 3dp):")
        for idx in top_idx:
            print(f"    food_after={vals[idx]:.3f}: {counts[idx]} occurrences")

        # The cap value should be the most common high food_after
        # Look for exact food_after values at high end
        high_food = food_a_np[food_a_np > 0.7]
        if len(high_food) > 50:
            rounded_high = np.round(high_food, 4)
            hvals, hcounts = np.unique(rounded_high, return_counts=True)
            htop = np.argsort(-hcounts)[:3]
            print(f"  Most common food_after > 0.7:")
            for idx in htop:
                print(f"    food_after={hvals[idx]:.4f}: {hcounts[idx]} occurrences")
            cap_value = hvals[htop[0]]
            winter_sev = 0.998 - cap_value
            print(f"  → Likely cap = 0.998 - winter_severity → winter_severity ≈ {winter_sev:.4f}")
        else:
            cap_value = None
            winter_sev = None

        # ── 2. Fit on LOW food (food_before < 0.3) to avoid cap ──────────
        mask_low = non_port & (food_b < 0.3) & (food_a > 0.001)
        d = arr[mask_low]
        fd = food_a[mask_low] - food_b[mask_low]
        n = len(d)

        if n < 50:
            print(f"  Only {n} low-food observations, skipping\n")
            continue

        print(f"\n  LOW food_before (<0.3): {n} observations")
        print(f"    food_delta: mean={fd.mean():.6f}, std={fd.std():.6f}")

        # Linear: intercept + food + pop + plains + forest
        X = np.column_stack([
            np.ones(n), d[:, 1], d[:, 0], d[:, 4], d[:, 5],
        ])
        c, r2, mae, maxe, res = fit_ols(X, fd)
        print(f"    int+food+pop+pl+fo: R²={r2:.6f}, MAE={mae:.6f}")
        print(f"      int={c[0]:.6f}, food={c[1]:.6f}, pop={c[2]:.6f}, "
              f"pl={c[3]:.6f}, fo={c[4]:.6f}")

        # + food²
        X2 = np.column_stack([
            np.ones(n), d[:, 1], d[:, 1]**2, d[:, 0], d[:, 4], d[:, 5],
        ])
        c2, r2_2, mae2, _, _ = fit_ols(X2, fd)
        print(f"    +food²:            R²={r2_2:.6f}, MAE={mae2:.6f}  [food²={c2[2]:.6f}]")

        # + mountain + settlement
        X3 = np.column_stack([
            np.ones(n), d[:, 1], d[:, 0], d[:, 4], d[:, 5], d[:, 6], d[:, 7],
        ])
        c3, r2_3, mae3, _, _ = fit_ols(X3, fd)
        print(f"    +mt+st:            R²={r2_3:.6f}, MAE={mae3:.6f}  "
              f"[mt={c3[5]:.6f}, st={c3[6]:.6f}]")

        # ── 3. Fit on MEDIUM food (0.3 < food_before < 0.6) ──────────────
        mask_med = non_port & (food_b >= 0.3) & (food_b < 0.6) & (food_a > 0.001)
        d_m = arr[mask_med]
        fd_m = food_a[mask_med] - food_b[mask_med]
        n_m = len(d_m)

        if n_m > 50:
            print(f"\n  MEDIUM food_before (0.3–0.6): {n_m} observations")
            X = np.column_stack([
                np.ones(n_m), d_m[:, 1], d_m[:, 0], d_m[:, 4], d_m[:, 5],
            ])
            c, r2, mae, maxe, res = fit_ols(X, fd_m)
            print(f"    int+food+pop+pl+fo: R²={r2:.6f}, MAE={mae:.6f}")
            print(f"      int={c[0]:.6f}, food={c[1]:.6f}, pop={c[2]:.6f}, "
                  f"pl={c[3]:.6f}, fo={c[4]:.6f}")

        # ── 4. Fit on HIGH food but BELOW cap ────────────────────────────
        if cap_value is not None:
            mask_high = non_port & (food_b >= 0.6) & (food_a < cap_value - 0.01) & (food_a > 0.001)
            d_h = arr[mask_high]
            fd_h = food_a[mask_high] - food_b[mask_high]
            n_h = len(d_h)

            if n_h > 50:
                print(f"\n  HIGH food_before (>0.6, below cap): {n_h} observations")
                X = np.column_stack([
                    np.ones(n_h), d_h[:, 1], d_h[:, 0], d_h[:, 4], d_h[:, 5],
                ])
                c, r2, mae, maxe, res = fit_ols(X, fd_h)
                print(f"    int+food+pop+pl+fo: R²={r2:.6f}, MAE={mae:.6f}")
                print(f"      int={c[0]:.6f}, food={c[1]:.6f}, pop={c[2]:.6f}, "
                      f"pl={c[3]:.6f}, fo={c[4]:.6f}")

        # ── 5. Combined: all non-capped observations ─────────────────────
        if cap_value is not None:
            mask_clean = non_port & (food_a < cap_value - 0.005) & (food_a > 0.001) & (food_b < 0.997)
        else:
            mask_clean = non_port & (food_a < 0.997) & (food_a > 0.001)

        d_c = arr[mask_clean]
        fd_c = food_a[mask_clean] - food_b[mask_clean]
        n_c = len(d_c)

        print(f"\n  ALL non-capped (cap-excluded): {n_c} observations")
        X = np.column_stack([
            np.ones(n_c), d_c[:, 1], d_c[:, 0], d_c[:, 4], d_c[:, 5],
        ])
        c, r2, mae, maxe, res = fit_ols(X, fd_c)
        print(f"    int+food+pop+pl+fo: R²={r2:.6f}, MAE={mae:.6f}")
        print(f"      int={c[0]:.6f}, food={c[1]:.6f}, pop={c[2]:.6f}, "
              f"pl={c[3]:.6f}, fo={c[4]:.6f}")

        # Still check food²
        X2 = np.column_stack([
            np.ones(n_c), d_c[:, 1], d_c[:, 1]**2, d_c[:, 0], d_c[:, 4], d_c[:, 5],
        ])
        c2, r2_2, mae2, _, _ = fit_ols(X2, fd_c)
        print(f"    +food²:            R²={r2_2:.6f}, MAE={mae2:.6f}  [food²={c2[2]:.6f}]")

        # ── 6. Residual distribution ─────────────────────────────────────
        print(f"    residual percentiles: p10={np.percentile(res, 10):.6f}, "
              f"p50={np.percentile(res, 50):.6f}, p90={np.percentile(res, 90):.6f}")

        print()


if __name__ == "__main__":
    main()
