"""
Wealth radius sweep: check how much of positive d_wealth is explained
by raids at increasing Chebyshev distances (1 through 15).
Also check: enemy *presence* (not just raids) at each radius.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_replays import load_all_replays, build_transition_df

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

replays = load_all_replays()
trans = build_transition_df(replays)

alive = trans[(trans["alive"]) & (trans["alive_next"])].copy()
alive["owner_same"] = alive["owner_id"] == alive["owner_id_next"]
clean = alive[(alive["d_defense"] >= -0.001) & alive["owner_same"]].copy()

pos = clean[clean["d_wealth"] > 1e-6].copy()
neg = clean[clean["d_wealth"] < -1e-6].copy()

# Build raid index: (round_id, seed, step) -> [(x, y, owner_id)]
raided_all = alive[alive["d_defense"] < -0.01]
raid_index = {}
for _, r in raided_all.iterrows():
    key = (r["round_id"], r["seed"], r["step"])
    if key not in raid_index:
        raid_index[key] = []
    raid_index[key].append((r["x"], r["y"], r["owner_id"]))

# Build enemy settlement index: (round_id, seed, step) -> [(x, y, owner_id)]
print("Building settlement index from replays...")
settlement_index = {}
for round_id, seed, frames in replays:
    for frame in frames:
        step = frame["step"]
        key = (round_id[:8], seed, step)
        slist = []
        for s in frame["settlements"]:
            if s["alive"]:
                slist.append((s["x"], s["y"], s["owner_id"]))
        settlement_index[key] = slist

def check_nearby(row, index, max_dist, filter_owner=None):
    """Check if any entry in index is within Chebyshev max_dist.
    filter_owner: 'enemy' = different owner, 'own' = same owner, None = any.
    """
    key = (row["round_id"], row["seed"], row["step"])
    if key not in index:
        return False
    for ix, iy, oid in index[key]:
        if ix == row["x"] and iy == row["y"]:
            continue
        if max(abs(ix - row["x"]), abs(iy - row["y"])) <= max_dist:
            if filter_owner == "enemy" and oid == row["owner_id"]:
                continue
            if filter_owner == "own" and oid != row["owner_id"]:
                continue
            return True
    return False

# ================================================================
# Sweep: raids at increasing radii
# ================================================================
print("Sweeping raid radius...")
radii = list(range(1, 16))
raid_coverage = []
enemy_raid_coverage = []
enemy_presence_coverage = []

for d in radii:
    # Any raid nearby
    r_any = pos.apply(lambda row, md=d: check_nearby(row, raid_index, md), axis=1)
    raid_coverage.append(r_any.mean() * 100)

    # Enemy raid nearby (raid on different owner)
    r_enemy = pos.apply(lambda row, md=d: check_nearby(row, raid_index, md, "enemy"), axis=1)
    enemy_raid_coverage.append(r_enemy.mean() * 100)

    # Enemy settlement nearby (not necessarily raided)
    e_near = pos.apply(lambda row, md=d: check_nearby(row, settlement_index, md, "enemy"), axis=1)
    enemy_presence_coverage.append(e_near.mean() * 100)

    print(f"  dist≤{d:2d}: raid={raid_coverage[-1]:.1f}%, enemy_raid={enemy_raid_coverage[-1]:.1f}%, enemy_present={enemy_presence_coverage[-1]:.1f}%")

# Also check baseline: what fraction of ZERO d_wealth settlements have these?
print("\nBaseline (zero d_wealth):")
zero_sample = clean[clean["d_wealth"].abs() <= 1e-6].sample(min(5000, len(clean)), random_state=42)
baseline_raid = []
baseline_enemy = []
for d in radii:
    r = zero_sample.apply(lambda row, md=d: check_nearby(row, raid_index, md), axis=1).mean() * 100
    e = zero_sample.apply(lambda row, md=d: check_nearby(row, settlement_index, md, "enemy"), axis=1).mean() * 100
    baseline_raid.append(r)
    baseline_enemy.append(e)
    print(f"  dist≤{d:2d}: raid={r:.1f}%, enemy_present={e:.1f}%")

# Same for negative d_wealth
print("\nNegative d_wealth:")
neg_raid_coverage = []
neg_enemy_coverage = []
for d in radii:
    r = neg.apply(lambda row, md=d: check_nearby(row, raid_index, md), axis=1).mean() * 100
    e = neg.apply(lambda row, md=d: check_nearby(row, settlement_index, md, "enemy"), axis=1).mean() * 100
    neg_raid_coverage.append(r)
    neg_enemy_coverage.append(e)
    print(f"  dist≤{d:2d}: raid={r:.1f}%, enemy_present={e:.1f}%")

# ================================================================
# Plotting
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Raid proximity coverage
axes[0].plot(radii, raid_coverage, "o-", label="Any raid", color="coral")
axes[0].plot(radii, enemy_raid_coverage, "s-", label="Enemy raid", color="red")
axes[0].plot(radii, baseline_raid, "x--", label="Baseline (d_w=0)", color="gray")
axes[0].plot(radii, neg_raid_coverage, "^--", label="Negative d_wealth", color="steelblue")
axes[0].set_xlabel("Chebyshev distance")
axes[0].set_ylabel("% with raid nearby")
axes[0].set_title("Raid proximity vs positive d_wealth")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 2. Enemy presence coverage
axes[1].plot(radii, enemy_presence_coverage, "o-", label="Positive d_wealth", color="coral")
axes[1].plot(radii, baseline_enemy, "x--", label="Baseline (d_w=0)", color="gray")
axes[1].plot(radii, neg_enemy_coverage, "^--", label="Negative d_wealth", color="steelblue")
axes[1].set_xlabel("Chebyshev distance")
axes[1].set_ylabel("% with enemy nearby")
axes[1].set_title("Enemy settlement proximity")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# 3. Excess over baseline (the signal)
excess_raid = [r - b for r, b in zip(raid_coverage, baseline_raid)]
excess_enemy = [e - b for e, b in zip(enemy_presence_coverage, baseline_enemy)]
axes[2].plot(radii, excess_raid, "o-", label="Raid excess", color="coral")
axes[2].plot(radii, excess_enemy, "s-", label="Enemy presence excess", color="red")
axes[2].set_xlabel("Chebyshev distance")
axes[2].set_ylabel("Excess % over baseline")
axes[2].set_title("Signal: excess over zero-d_wealth baseline")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "wealth_radius.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "wealth_radius.png"), dpi=150)
print(f"\nSaved: {OUT}/wealth_radius.pdf")
