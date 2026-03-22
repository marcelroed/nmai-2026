"""
Final verification of spawn formulas.

Hypotheses to test:
1. Ruin reclaim: food = nearest_food * 0.20, wealth = nearest_wealth * 0.20
2. Growth: food = parent_food * 0.20 (at spawn time), wealth = parent_wealth * 0.20

For growth food: ratio to prev frame should be ~0.20 * (1 + food_delta/parent_food).
Test by computing: child_food / (parent_food_prev + estimated_food_delta) ≈ 0.20

Also test parent_cost (pop deduction from parent).

Reproduce: python3 data-analysis/spawn_stats4.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RUIN = 3
SETTLEMENT = 1
PORT = 2
PLAINS = 0
FOREST = 4
MOUNTAIN = 5
OCEAN = 6


def load_all_replays():
    replays = []
    for rd in sorted(DATA_DIR.iterdir()):
        if not rd.is_dir():
            continue
        ad = rd / "analysis"
        if not ad.exists():
            continue
        for f in sorted(ad.glob("replay_seed_index=*.json")):
            with open(f) as fh:
                data = json.load(fh)
            replays.append(data)
    return replays


def count_neighbors(grid, x, y, height, width):
    """Count 8-connected neighbors by type."""
    n_pl = n_fo = n_mt = n_st = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                t = grid[ny][nx]
                if t == PLAINS or t == 0:  # Plains or Empty
                    n_pl += 1
                elif t == FOREST:
                    n_fo += 1
                elif t == MOUNTAIN:
                    n_mt += 1
                elif t in (SETTLEMENT, PORT, RUIN):
                    n_st += 1
    return n_pl, n_fo, n_mt, n_st


def estimate_food_delta(food, pop, n_pl, n_fo, n_mt, n_st):
    """Estimate food delta using the known logistic model."""
    omf = 1.0 - food
    delta = (0.043 * n_pl * omf
             + 0.074 * n_fo * omf
             + 0.0 * n_mt * omf
             - 0.017 * n_st * omf
             + 0.91 * food * omf
             - 0.113 * pop)
    return delta


def main():
    replays = load_all_replays()

    # ── Ruin reclaim: test wealth = nearest_wealth * 0.20 ────────────
    ruin_child_wealth = []
    ruin_nearest_wealth_same = []
    ruin_child_food = []
    ruin_nearest_food_same = []

    # ── Growth: test food = parent_food_at_spawn * 0.20 ─────────────
    growth_child_food = []
    growth_parent_food_prev = []
    growth_parent_food_estimated = []  # prev + food_delta
    growth_parent_food_same = []
    growth_child_wealth = []
    growth_parent_wealth_prev = []
    growth_parent_pop_loss = []  # pop_prev - pop_same (= parent_cost)

    for replay in replays:
        frames = replay["frames"]
        height = len(frames[0]["grid"])
        width = len(frames[0]["grid"][0])

        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]

            before_map = {}
            for s in fb["settlements"]:
                if s["alive"]:
                    before_map[(s["x"], s["y"])] = s

            after_map = {}
            for s in fa["settlements"]:
                if s["alive"]:
                    after_map[(s["x"], s["y"])] = s

            for s in fa["settlements"]:
                if not s["alive"]:
                    continue
                pos = (s["x"], s["y"])
                if pos in before_map:
                    continue

                x, y = pos
                was_ruin = fb["grid"][y][x] == RUIN
                owner = s["owner_id"]

                if was_ruin:
                    # ── RUIN RECLAIM ──
                    # Find nearest same-owner in SAME frame
                    min_dist = float("inf")
                    nearest = None
                    for apos, a_s in after_map.items():
                        if apos == pos or a_s["owner_id"] != owner:
                            continue
                        ax, ay = apos
                        dx, dy = ax - x, ay - y
                        dist = (dx*dx + dy*dy) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest = a_s

                    if nearest is not None:
                        ruin_child_food.append(s["food"])
                        ruin_nearest_food_same.append(nearest["food"])
                        ruin_child_wealth.append(s["wealth"])
                        ruin_nearest_wealth_same.append(nearest["wealth"])

                else:
                    # ── GROWTH ──
                    # Find parent: same owner, existed before, pop decreased
                    best_parent_prev = None
                    best_parent_same = None
                    min_dist = float("inf")

                    for bpos, bs in before_map.items():
                        if bs["owner_id"] != owner:
                            continue
                        if bpos not in after_map:
                            continue
                        a_s = after_map[bpos]
                        if a_s["population"] >= bs["population"]:
                            continue
                        bx, by = bpos
                        dx, dy = bx - x, by - y
                        dist = (dx*dx + dy*dy) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_parent_prev = bs
                            best_parent_same = a_s

                    if best_parent_prev is None or best_parent_prev["food"] < 0.05:
                        continue

                    # Estimate parent food at spawn time (after phase 1)
                    px, py = best_parent_prev["x"], best_parent_prev["y"]
                    n_pl, n_fo, n_mt, n_st = count_neighbors(
                        fb["grid"], px, py, height, width)
                    food_delta = estimate_food_delta(
                        best_parent_prev["food"], best_parent_prev["population"],
                        n_pl, n_fo, n_mt, n_st)
                    parent_food_at_spawn = best_parent_prev["food"] + food_delta

                    growth_child_food.append(s["food"])
                    growth_parent_food_prev.append(best_parent_prev["food"])
                    growth_parent_food_estimated.append(parent_food_at_spawn)
                    growth_parent_food_same.append(best_parent_same["food"])
                    growth_child_wealth.append(s["wealth"])
                    growth_parent_wealth_prev.append(best_parent_prev["wealth"])

                    # Parent pop loss = parent_cost
                    pop_loss = best_parent_prev["population"] - best_parent_same["population"]
                    growth_parent_pop_loss.append(pop_loss)

    # Convert
    ruin_child_food = np.array(ruin_child_food)
    ruin_nearest_food_same = np.array(ruin_nearest_food_same)
    ruin_child_wealth = np.array(ruin_child_wealth)
    ruin_nearest_wealth_same = np.array(ruin_nearest_wealth_same)

    growth_child_food = np.array(growth_child_food)
    growth_parent_food_prev = np.array(growth_parent_food_prev)
    growth_parent_food_estimated = np.array(growth_parent_food_estimated)
    growth_parent_food_same = np.array(growth_parent_food_same)
    growth_child_wealth = np.array(growth_child_wealth)
    growth_parent_wealth_prev = np.array(growth_parent_wealth_prev)
    growth_parent_pop_loss = np.array(growth_parent_pop_loss)

    # ═══════════════════════════════════════════════════════════════════
    # RUIN RECLAIM — wealth
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("RUIN RECLAIM — wealth = nearest_wealth * factor?")
    print("=" * 70)

    has_w = ruin_nearest_wealth_same > 0.005
    print(f"\n  Total ruin reclaims: {len(ruin_child_wealth)}")
    print(f"  With nearest_wealth > 0.005: {has_w.sum()}")

    if has_w.sum() > 0:
        ratios_w = ruin_child_wealth[has_w] / ruin_nearest_wealth_same[has_w]
        print(f"\n  child_wealth / nearest_wealth (same frame):")
        print(f"    mean={ratios_w.mean():.6f}  median={np.median(ratios_w):.6f}")
        print(f"    std={ratios_w.std():.6f}")
        for r in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
            exact = np.mean(np.abs(ratios_w - r) < 0.001) * 100
            mae = np.mean(np.abs(ratios_w - r))
            print(f"    ratio={r:.2f}: MAE={mae:.4f}  exact(±0.001)={exact:.1f}%")

    # Check: how many ruin reclaims have wealth == 0?
    zero_w = np.sum(ruin_child_wealth < 0.0005)
    print(f"\n  child_wealth == 0: {zero_w} ({100*zero_w/len(ruin_child_wealth):.1f}%)")
    print(f"  nearest_wealth == 0 (< 0.005): {(~has_w).sum()}")

    # For zero-child-wealth cases, what was nearest wealth?
    zero_mask = ruin_child_wealth < 0.0005
    if zero_mask.sum() > 0:
        nw = ruin_nearest_wealth_same[zero_mask]
        print(f"  When child_wealth=0: nearest_wealth mean={nw.mean():.4f}, "
              f"fraction<0.005={np.mean(nw<0.005):.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # GROWTH — food = parent_food_at_spawn * 0.20
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("GROWTH — food = parent_food_at_spawn * factor?")
    print("=" * 70)
    print(f"\n  N = {len(growth_child_food)}")

    # Using estimated parent food at spawn (prev + food_delta)
    valid = growth_parent_food_estimated > 0.05
    if valid.sum() > 0:
        ratios_est = growth_child_food[valid] / growth_parent_food_estimated[valid]
        print(f"\n  child_food / (parent_food_prev + food_delta):")
        print(f"    mean={ratios_est.mean():.6f}  median={np.median(ratios_est):.6f}")
        print(f"    std={ratios_est.std():.6f}")
        for r in [0.10, 0.15, 0.20, 0.25, 0.30]:
            exact = np.mean(np.abs(ratios_est - r) < 0.002) * 100
            mae = np.mean(np.abs(ratios_est - r))
            print(f"    ratio={r:.2f}: MAE={mae:.6f}  exact(±0.002)={exact:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # GROWTH — parent_cost (pop deduction)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("GROWTH — parent pop loss (parent_cost)")
    print("=" * 70)

    # The pop loss between frames includes: pop_growth (phase 2) - parent_cost (phase 3)
    # So raw pop loss = parent_cost - pop_growth
    # pop_growth depends on the population growth model
    print(f"\n  Raw pop loss (prev - same frame):")
    print(f"    mean={growth_parent_pop_loss.mean():.6f}  median={np.median(growth_parent_pop_loss):.6f}")
    print(f"    std={growth_parent_pop_loss.std():.6f}")
    # This is net loss including pop growth, so actual parent_cost is higher
    # parent_cost = raw_pop_loss + pop_growth ≈ raw_pop_loss + pop_growth_rate * pop * ...
    vals, counts = np.unique(np.round(growth_parent_pop_loss, 3), return_counts=True)
    top = np.argsort(-counts)[:15]
    print(f"    top values:")
    for j in top:
        print(f"      {vals[j]:.3f}  ({counts[j]} times, {100*counts[j]/len(growth_parent_pop_loss):.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # INITIAL STATS (step 0)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("INITIAL STATS (step 0)")
    print("=" * 70)

    init_stats = {"food": [], "wealth": [], "pop": [], "defense": []}
    for replay in replays:
        f0 = replay["frames"][0]
        for s in f0["settlements"]:
            if s["alive"]:
                init_stats["food"].append(s["food"])
                init_stats["wealth"].append(s["wealth"])
                init_stats["pop"].append(s["population"])
                init_stats["defense"].append(s["defense"])

    for key, vals_list in init_stats.items():
        v = np.array(vals_list)
        print(f"\n  {key}: mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")
        # Check if uniform
        r = v.max() - v.min()
        expected_mean = (v.max() + v.min()) / 2
        expected_std = r / (12**0.5)
        print(f"    If Uniform[{v.min():.3f}, {v.max():.3f}]: "
              f"expected_mean={expected_mean:.4f} expected_std={expected_std:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Ruin wealth ratio histogram
    ax = axes[0, 0]
    if has_w.sum() > 0:
        ax.hist(ratios_w, bins=100, density=True, color="orange", alpha=0.7)
        ax.axvline(0.20, color="r", lw=2, label="0.20")
        ax.axvline(0.0, color="k", lw=1, ls="--", label="0.00")
    ax.set_xlabel("child_wealth / nearest_wealth (same frame)")
    ax.set_title("Ruin reclaim: wealth ratio")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.1, 1.0)

    # 2. Ruin wealth scatter
    ax = axes[0, 1]
    ax.scatter(ruin_nearest_wealth_same, ruin_child_wealth, s=2, alpha=0.1,
               color="orange", rasterized=True)
    lim = max(ruin_nearest_wealth_same.max(), 0.5)
    ax.plot([0, lim], [0, lim*0.20], "r-", lw=2, label="y=0.20x")
    ax.plot([0, lim], [0, 0], "k--", lw=1, label="y=0")
    ax.set_xlabel("nearest wealth (same frame)")
    ax.set_ylabel("child wealth")
    ax.set_title("Ruin: child wealth vs nearest wealth")
    ax.legend(fontsize=8)

    # 3. Growth food ratio using estimated parent food
    ax = axes[0, 2]
    if valid.sum() > 0:
        ax.hist(ratios_est, bins=100, density=True, color="steelblue", alpha=0.7)
        ax.axvline(0.20, color="r", lw=2, label="0.20")
        ax.axvline(0.10, color="k", lw=1.5, ls="--", label="0.10")
    ax.set_xlabel("child_food / (parent_food + food_delta)")
    ax.set_title("Growth: food ratio (est. parent food at spawn)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.8)

    # 4. Growth food scatter with estimated parent
    ax = axes[1, 0]
    if valid.sum() > 0:
        ax.scatter(growth_parent_food_estimated[valid], growth_child_food[valid],
                   s=2, alpha=0.1, color="steelblue", rasterized=True)
        lim = growth_parent_food_estimated[valid].max() * 1.05
        ax.plot([0, lim], [0, lim*0.20], "r-", lw=2, label="y=0.20x")
        ax.plot([0, lim], [0, lim*0.10], "k--", lw=1.5, label="y=0.10x")
    ax.set_xlabel("parent food (estimated at spawn)")
    ax.set_ylabel("child food")
    ax.set_title("Growth: child food vs parent food (at spawn)")
    ax.legend(fontsize=8)

    # 5. Pop loss histogram
    ax = axes[1, 1]
    ax.hist(growth_parent_pop_loss, bins=80, density=True, color="steelblue", alpha=0.7)
    ax.set_xlabel("parent pop loss (prev - same frame)")
    ax.set_title("Growth: parent pop loss")

    # 6. Initial stats distributions
    ax = axes[1, 2]
    for key, color in [("food", "green"), ("wealth", "gold"),
                        ("pop", "blue"), ("defense", "red")]:
        v = np.array(init_stats[key])
        ax.hist(v, bins=50, density=True, alpha=0.5, color=color, label=key)
    ax.set_xlabel("value")
    ax.set_title("Initial stats (step 0)")
    ax.legend(fontsize=8)

    fig.suptitle("Spawn stats — final verification", fontsize=14)
    plt.tight_layout()
    fig.savefig("spawn_stats4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved spawn_stats4.png")


if __name__ == "__main__":
    main()
