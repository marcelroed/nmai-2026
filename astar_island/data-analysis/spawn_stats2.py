"""
Refined spawn analysis: identify actual parent by pop decrease.

Growth phase: parent.pop -= parent_cost, child spawns nearby.
So the actual parent is the nearby same-owner settlement whose pop decreased.

Reproduce:  python3 data-analysis/spawn_stats2.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RUIN = 3


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


def main():
    replays = load_all_replays()

    # Growth spawns: identify parent by pop decrease
    growth_records = []
    ruin_records = []

    for replay in replays:
        frames = replay["frames"]
        rid = replay["round_id"]
        seed = replay["seed_index"]

        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            step = fa["step"]
            grid_before = fb["grid"]

            # Build maps
            before_map = {}
            for s in fb["settlements"]:
                if s["alive"]:
                    before_map[(s["x"], s["y"])] = s

            after_map = {}
            for s in fa["settlements"]:
                if s["alive"]:
                    pos = (s["x"], s["y"])
                    if pos not in after_map:
                        after_map[pos] = s

            before_positions = set(before_map.keys())

            # Find new settlements
            for pos, sa in after_map.items():
                if pos in before_positions:
                    continue

                x, y = pos
                was_ruin = grid_before[y][x] == RUIN

                # Find candidate parents: same owner, nearby, pop decreased
                owner = sa["owner_id"]
                candidates = []
                for bpos, sb in before_map.items():
                    if sb["owner_id"] != owner:
                        continue
                    # Chebyshev distance
                    dx = abs(bpos[0] - x)
                    dy = abs(bpos[1] - y)
                    cheb = max(dx, dy)
                    if cheb > 5:  # generous range
                        continue

                    # Check if this settlement exists in after frame
                    sa_parent = after_map.get(bpos)
                    if sa_parent is None:
                        continue
                    if sa_parent["owner_id"] != owner:
                        continue

                    pop_loss = sb["population"] - sa_parent["population"]
                    candidates.append({
                        "pos": bpos,
                        "food_before": sb["food"],
                        "wealth_before": sb["wealth"],
                        "pop_before": sb["population"],
                        "pop_after": sa_parent["population"],
                        "pop_loss": pop_loss,
                        "cheb_dist": cheb,
                    })

                # Best parent: nearby + lost pop
                parent = None
                if candidates:
                    # Prefer candidates that lost pop, closest
                    lost_pop = [c for c in candidates if c["pop_loss"] > 0.05]
                    if lost_pop:
                        parent = min(lost_pop, key=lambda c: c["cheb_dist"])
                    else:
                        parent = min(candidates, key=lambda c: c["cheb_dist"])

                record = {
                    "food": sa["food"],
                    "wealth": sa["wealth"],
                    "pop": sa["population"],
                    "defense": sa["defense"],
                    "is_port": sa.get("has_port", False),
                    "step": step,
                    "was_ruin": was_ruin,
                    "parent_food": parent["food_before"] if parent else None,
                    "parent_wealth": parent["wealth_before"] if parent else None,
                    "parent_pop_loss": parent["pop_loss"] if parent else None,
                    "parent_dist": parent["cheb_dist"] if parent else None,
                    "round": rid,
                }

                if was_ruin:
                    ruin_records.append(record)
                else:
                    growth_records.append(record)

    print(f"Growth spawns: {len(growth_records)}")
    print(f"Ruin reclaims: {len(ruin_records)}")

    # ═══════════════════════════════════════════════════════════════════
    # GROWTH SPAWNS — food ratio
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("GROWTH SPAWNS — food at spawn")
    print(f"{'='*70}")

    # Filter to confirmed parents (pop decreased)
    confirmed = [r for r in growth_records
                 if r["parent_pop_loss"] is not None and r["parent_pop_loss"] > 0.05]
    print(f"\n  Confirmed parent (pop decreased): {len(confirmed)}")

    if confirmed:
        child_food = np.array([r["food"] for r in confirmed])
        parent_food = np.array([r["parent_food"] for r in confirmed])
        ratios = child_food / np.maximum(parent_food, 1e-6)

        # Filter out crazy ratios (parent food near 0)
        good = parent_food > 0.05
        ratios_good = ratios[good]
        child_good = child_food[good]
        parent_good = parent_food[good]

        print(f"  With parent_food > 0.05: {good.sum()}")
        print(f"\n  child_food / parent_food:")
        print(f"    mean={ratios_good.mean():.6f}  std={ratios_good.std():.6f}")
        print(f"    median={np.median(ratios_good):.6f}")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"    P{p:02d} = {np.percentile(ratios_good, p):.6f}")

        # Check common exact ratios
        print(f"\n  Testing exact ratio hypotheses:")
        for test_ratio in [0.1, 0.15, 0.2, 0.25, 0.3]:
            expected = parent_good * test_ratio
            err = np.abs(child_good - expected)
            pct_exact = np.mean(err < 0.0015) * 100
            mae = np.mean(err)
            print(f"    ratio={test_ratio:.2f}: MAE={mae:.6f}  "
                  f"exact(±0.001)={pct_exact:.1f}%")

        # parent pop loss (= parent_cost)
        pop_loss = np.array([r["parent_pop_loss"] for r in confirmed])
        print(f"\n  Parent pop loss (= parent_cost):")
        vals, counts = np.unique(np.round(pop_loss, 4), return_counts=True)
        top = np.argsort(-counts)[:10]
        for j in top:
            print(f"    {vals[j]:.4f}  ({counts[j]} times, {100*counts[j]/len(confirmed):.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # GROWTH SPAWNS — wealth
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("GROWTH SPAWNS — wealth at spawn")
    print(f"{'='*70}")

    if confirmed:
        child_wealth = np.array([r["wealth"] for r in confirmed])
        parent_wealth = np.array([r["parent_wealth"] for r in confirmed])

        print(f"\n  child_wealth:")
        print(f"    mean={child_wealth.mean():.6f}  std={child_wealth.std():.6f}")
        print(f"    min={child_wealth.min():.6f}  max={child_wealth.max():.6f}")
        vals, counts = np.unique(np.round(child_wealth, 4), return_counts=True)
        top = np.argsort(-counts)[:10]
        print(f"    top values:")
        for j in top:
            print(f"      {vals[j]:.4f}  ({counts[j]} times, {100*counts[j]/len(confirmed):.1f}%)")

        # Check if wealth = 0 always
        print(f"\n    wealth == 0: {np.sum(child_wealth < 0.0005)} "
              f"({100*np.mean(child_wealth < 0.0005):.1f}%)")

        # Check if wealth = parent_wealth * factor
        good_w = parent_wealth > 0.01
        if good_w.sum() > 10:
            w_ratio = child_wealth[good_w] / parent_wealth[good_w]
            print(f"\n  child_wealth / parent_wealth (where parent > 0.01):")
            print(f"    mean={w_ratio.mean():.6f}  median={np.median(w_ratio):.6f}")
            for test_ratio in [0.0, 0.05, 0.1, 0.2]:
                expected = parent_wealth[good_w] * test_ratio
                err = np.abs(child_wealth[good_w] - expected)
                pct_exact = np.mean(err < 0.0015) * 100
                print(f"    ratio={test_ratio:.2f}: exact(±0.001)={pct_exact:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # RUIN RECLAIMS — food and wealth
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("RUIN RECLAIMS — food and wealth at spawn")
    print(f"{'='*70}")

    if ruin_records:
        has_parent = [r for r in ruin_records if r["parent_food"] is not None]
        child_food = np.array([r["food"] for r in has_parent])
        parent_food = np.array([r["parent_food"] for r in has_parent])
        child_wealth = np.array([r["wealth"] for r in has_parent])
        parent_wealth = np.array([r["parent_wealth"] for r in has_parent])

        good = parent_food > 0.05
        print(f"\n  Ruin spawns with nearby parent (food > 0.05): {good.sum()}")

        if good.sum() > 0:
            ratios = child_food[good] / parent_food[good]
            print(f"\n  child_food / nearest_food:")
            print(f"    mean={ratios.mean():.6f}  median={np.median(ratios):.6f}")
            for p in [5, 25, 50, 75, 95]:
                print(f"    P{p:02d} = {np.percentile(ratios, p):.6f}")

            for test_ratio in [0.1, 0.15, 0.2, 0.25, 0.3]:
                expected = parent_food[good] * test_ratio
                err = np.abs(child_food[good] - expected)
                pct_exact = np.mean(err < 0.0015) * 100
                mae = np.mean(err)
                print(f"    ratio={test_ratio:.2f}: MAE={mae:.6f}  exact(±0.001)={pct_exact:.1f}%")

        # Wealth for ruin spawns
        print(f"\n  Ruin wealth:")
        print(f"    mean={child_wealth.mean():.6f}  std={child_wealth.std():.6f}")
        vals, counts = np.unique(np.round(child_wealth, 4), return_counts=True)
        top = np.argsort(-counts)[:5]
        for j in top:
            print(f"    {vals[j]:.4f}  ({counts[j]} times, {100*counts[j]/len(has_parent):.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 0: Initial settlements
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STEP 0: Initial settlement stats")
    print(f"{'='*70}")

    init_food = []
    init_wealth = []
    init_pop = []
    init_defense = []

    for replay in replays:
        frame0 = replay["frames"][0]
        for s in frame0["settlements"]:
            if s["alive"]:
                init_food.append(s["food"])
                init_wealth.append(s["wealth"])
                init_pop.append(s["population"])
                init_defense.append(s["defense"])

    init_food = np.array(init_food)
    init_wealth = np.array(init_wealth)
    init_pop = np.array(init_pop)
    init_defense = np.array(init_defense)

    for name, vals in [("food", init_food), ("wealth", init_wealth),
                       ("pop", init_pop), ("defense", init_defense)]:
        print(f"\n  {name}:")
        print(f"    mean={vals.mean():.6f}  std={vals.std():.6f}")
        print(f"    min={vals.min():.6f}  max={vals.max():.6f}")
        uv, uc = np.unique(np.round(vals, 4), return_counts=True)
        top = np.argsort(-uc)[:5]
        for j in top:
            print(f"    {uv[j]:.4f}  ({uc[j]} times, {100*uc[j]/len(vals):.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # PLOT
    # ═══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Growth: child food vs parent food (confirmed parents only)
    ax = axes[0, 0]
    if confirmed:
        cf = np.array([r["food"] for r in confirmed])
        pf = np.array([r["parent_food"] for r in confirmed])
        good = pf > 0.01
        ax.scatter(pf[good], cf[good], s=3, alpha=0.15, color="steelblue", rasterized=True)
        lim = 1.0
        ax.plot([0, lim], [0, lim*0.1], "r--", lw=1.5, label="y = 0.1x")
        ax.plot([0, lim], [0, lim*0.2], "g--", lw=1.5, label="y = 0.2x")
        ax.set_xlabel("parent food")
        ax.set_ylabel("child food")
        ax.set_title("Growth: child food = ? × parent food")
        ax.legend(fontsize=8)

    # Growth: food ratio histogram
    ax = axes[0, 1]
    if confirmed:
        good = np.array([r["parent_food"] for r in confirmed]) > 0.05
        ratios_g = cf[good] / pf[good]
        ratios_g = ratios_g[ratios_g < 0.6]
        ax.hist(ratios_g, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
        for r, c in [(0.1, "red"), (0.2, "green")]:
            ax.axvline(r, color=c, lw=2, ls="--", label=f"ratio={r}")
        ax.set_xlabel("child_food / parent_food")
        ax.set_ylabel("count")
        ax.set_title("Growth: food ratio distribution")
        ax.legend(fontsize=8)

    # Ruin: food ratio histogram
    ax = axes[0, 2]
    if ruin_records:
        has_p = [r for r in ruin_records if r["parent_food"] is not None]
        cf_r = np.array([r["food"] for r in has_p])
        pf_r = np.array([r["parent_food"] for r in has_p])
        good = pf_r > 0.05
        ratios_r = cf_r[good] / pf_r[good]
        ratios_r = ratios_r[ratios_r < 0.6]
        ax.hist(ratios_r, bins=100, color="orange", alpha=0.7, edgecolor="white")
        for r, c in [(0.1, "red"), (0.2, "green")]:
            ax.axvline(r, color=c, lw=2, ls="--", label=f"ratio={r}")
        ax.set_xlabel("spawn_food / nearest_food")
        ax.set_ylabel("count")
        ax.set_title("Ruin reclaim: food ratio distribution")
        ax.legend(fontsize=8)

    # Growth: wealth histogram
    ax = axes[1, 0]
    if confirmed:
        cw = np.array([r["wealth"] for r in confirmed])
        ax.hist(cw, bins=100, range=(0, 0.15), color="steelblue", alpha=0.7)
        ax.set_xlabel("child wealth at spawn")
        ax.set_ylabel("count")
        ax.set_title("Growth: wealth at spawn")

    # Ruin: wealth histogram
    ax = axes[1, 1]
    if ruin_records:
        rw = np.array([r["wealth"] for r in ruin_records])
        ax.hist(rw, bins=100, range=(0, 0.05), color="orange", alpha=0.7)
        ax.set_xlabel("spawn wealth")
        ax.set_ylabel("count")
        ax.set_title("Ruin reclaim: wealth at spawn")

    # Initial: food and wealth histograms
    ax = axes[1, 2]
    ax.hist(init_food, bins=50, alpha=0.6, color="steelblue", label="food", density=True)
    ax.hist(init_wealth, bins=50, alpha=0.6, color="orange", label="wealth", density=True)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title("Step 0: initial food and wealth")
    ax.legend(fontsize=8)

    fig.suptitle("Settlement spawn: food & wealth distributions", fontsize=14)
    plt.tight_layout()
    fig.savefig("spawn_stats2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved spawn_stats2.png")


if __name__ == "__main__":
    main()
