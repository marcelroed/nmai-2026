"""
Food analysis v3: diagnose capping, find winter_severity, then fit clean model.

Reproduce:  python3 data-analysis/analyze_food_v3.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN, OCEAN, PLAINS = 0, 1, 2, 3, 4, 5, 10, 11


def terrain_adj(grid, x, y):
    h, w = len(grid), len(grid[0])
    np_, nf, nm, ns = 0, 0, 0, 0
    for ny in range(max(0, y - 1), min(h, y + 2)):
        for nx in range(max(0, x - 1), min(w, x + 2)):
            if nx == x and ny == y:
                continue
            t = grid[ny][nx]
            if t in (PLAINS, EMPTY):
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
    rows = []
    for replay in replays:
        frames = replay["frames"]
        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            grid = fb["grid"]
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
                if sa["defense"] - sb["defense"] < -0.001:
                    continue
                x, y = pos
                adj = terrain_adj(grid, x, y)
                is_port = 1 if grid[y][x] == PORT else 0
                rows.append((
                    sb["population"], sb["food"], sb["wealth"],
                    sa["population"], sa["food"],
                    *adj,
                    is_port,
                ))
    return rows


def fit_ols(X, y):
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    res = y - pred
    mae = np.mean(np.abs(res))
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else 0
    return coeffs, r2, mae, res


def main():
    rounds = load_replays_by_round()
    print(f"Loaded {sum(len(v) for v in rounds.values())} replays from "
          f"{len(rounds)} rounds\n")

    for rid in sorted(rounds):
        replays = rounds[rid]
        raw = collect_rows(replays)
        arr = np.array(raw, dtype=np.float64)

        non_port = arr[:, 9] == 0
        d = arr[non_port]
        food_b = d[:, 1]
        food_a = d[:, 4]
        pop_b = d[:, 0]
        n_pl, n_fo, n_mt, n_st = d[:, 5], d[:, 6], d[:, 7], d[:, 8]

        print(f"{'=' * 72}")
        print(f"Round {rid[:12]}  ({len(d)} non-port obs)")
        print(f"{'=' * 72}")

        # ── 1. Max food values ─────────────────────────────────────────────
        print(f"  food_before: max={food_b.max():.6f}")
        print(f"  food_after:  max={food_a.max():.6f}")

        # ── 2. Find winter_severity from capped cases ──────────────────────
        # Settlements with good terrain, low pop should produce max food_delta.
        # If food_before + true_delta > cap, then food_after = cap - winter.
        # Group high-food-after values and find the cluster.
        # For food_before > 0.8 with good terrain, food_after should be capped.
        good_terrain = (n_pl + n_fo >= 4) & (pop_b < 0.8) & (food_b > 0.3)
        if good_terrain.sum() > 20:
            fa_gt = food_a[good_terrain]
            print(f"\n  Capped-case detection (good terrain, low pop, n={good_terrain.sum()}):")
            print(f"    food_after max={fa_gt.max():.6f}, p99={np.percentile(fa_gt, 99):.6f}, "
                  f"p95={np.percentile(fa_gt, 95):.6f}")

            # The maximum food_after among these should be exactly cap - winter
            cap_minus_winter = fa_gt.max()
            # Check how many are at this max (within tolerance)
            at_cap = np.sum(np.abs(fa_gt - cap_minus_winter) < 0.001)
            print(f"    values within 0.001 of max: {at_cap}")
            print(f"    → cap - winter ≈ {cap_minus_winter:.6f}")

        # ── 3. Try to determine cap vs winter separately ───────────────────
        # For settlements with VERY high food_before and still producing food,
        # the cap limits food_after = cap - winter regardless of food_before.
        # So food_delta = (cap - winter) - food_before for capped cases.
        # This means food_delta + food_before = cap - winter for all capped cases.
        food_sum = food_a  # food_after = food_before + food_delta = food_sum
        # For capped: food_sum = cap - winter (constant)
        # For uncapped: food_sum = food_before + true_delta (varies)

        # Check if there's a ceiling in food_after
        top_food_a = np.sort(food_a)[-min(100, len(food_a)):]
        print(f"\n  Top food_after values: {top_food_a[-5:]}")
        print(f"    If cap=0.998: winter = {0.998 - food_a.max():.6f}")
        print(f"    If cap=1.000: winter = {1.000 - food_a.max():.6f}")

        # ── 4. Binned analysis: food_delta vs food_before ──────────────────
        print(f"\n  Binned food_delta by food_before (pop in [0.8, 1.2], "
              f"n_pl+n_fo in [3,5]):")
        mask_ctrl = (pop_b >= 0.8) & (pop_b <= 1.2) & (n_pl + n_fo >= 3) & (n_pl + n_fo <= 5)
        if mask_ctrl.sum() > 100:
            fb_ctrl = food_b[mask_ctrl]
            fd_ctrl = food_a[mask_ctrl] - fb_ctrl
            bins = np.arange(0, 1.0, 0.1)
            for lo in bins:
                hi = lo + 0.1
                in_bin = (fb_ctrl >= lo) & (fb_ctrl < hi)
                if in_bin.sum() >= 10:
                    fd_bin = fd_ctrl[in_bin]
                    print(f"    food_b [{lo:.1f},{hi:.1f}): n={in_bin.sum():4d}, "
                          f"mean_delta={fd_bin.mean():>8.4f}, "
                          f"std={fd_bin.std():>7.4f}, "
                          f"mean_food_a={food_a[mask_ctrl][in_bin].mean():>7.4f}")

        # ── 5. Food_after ceiling: what's the precise max? ─────────────────
        # Check if food_after ever equals 0.998
        exact_998 = np.sum(np.abs(food_a - 0.998) < 0.0005)
        exact_100 = np.sum(np.abs(food_a - 1.0) < 0.0005)
        print(f"\n  food_after = 0.998 (±0.0005): {exact_998} cases")
        print(f"  food_after = 1.000 (±0.0005): {exact_100} cases")

        # ── 6. Try fitting food_after directly (not delta) ─────────────────
        # If capping prevents us from seeing the delta, maybe we can fit:
        # food_after = f(food_before, pop, terrain, ...) and detect the cap as
        # a ceiling in the prediction.
        # For uncapped: food_after = food_before + delta = food_before + base + ...
        # For capped: food_after = cap - winter
        # Let's try: food_after = a*food_before + b*pop + c*n_pl + d*n_fo + e
        # on LOW food cases only
        mask_low = (food_b < 0.25) & (food_a > 0.001)
        d_low = d[mask_low]
        if len(d_low) > 50:
            X = np.column_stack([
                np.ones(len(d_low)),
                d_low[:, 1],  # food_b
                d_low[:, 0],  # pop_b
                d_low[:, 5],  # n_pl
                d_low[:, 6],  # n_fo
            ])
            c_fit, r2, mae, res = fit_ols(X, d_low[:, 4])
            print(f"\n  Fit food_after (food_b < 0.25): R²={r2:.6f}, MAE={mae:.6f}")
            print(f"    food_after = {c_fit[0]:.6f} + {c_fit[1]:.6f}*food_b "
                  f"+ {c_fit[2]:.6f}*pop + {c_fit[3]:.6f}*n_pl + {c_fit[4]:.6f}*n_fo")
            print(f"    → food_coeff = {c_fit[1]:.6f} means food_b contributes "
                  f"{c_fit[1]:.4f} to food_after (1.0 = no feedback)")

        print()


if __name__ == "__main__":
    main()
