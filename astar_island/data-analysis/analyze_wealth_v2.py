"""
Wealth mechanics v2: deeper dive into what drives wealth changes.

From v1 we know:
- 83.5% of non-raided same-owner transitions have d_wealth exactly 0
- 38.9% of positive d_wealth explained by raids at Chebyshev dist ≤ 2
- 82.8% explained at dist ≤ 5

New questions:
- What explains the remaining ~17% of positive d_wealth not near raids?
- Is there a trade mechanism (ports, same-owner neighbors)?
- What drives negative d_wealth changes?
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

# Non-raided, same-owner
clean = alive[(alive["d_defense"] >= -0.001) & alive["owner_same"]].copy()

# Build raid index
raided_all = alive[alive["d_defense"] < -0.01]
raid_index = {}
for _, r in raided_all.iterrows():
    key = (r["round_id"], r["seed"], r["step"])
    if key not in raid_index:
        raid_index[key] = []
    raid_index[key].append((r["x"], r["y"], r["owner_id"]))

# ================================================================
# Build a richer per-settlement context from raw replay data
# Need: count of same-owner neighbors, count of enemy neighbors,
#       whether this is a "border" settlement, nearby port count
# ================================================================
print("Building settlement neighbor context from replays...")

# We need to know all settlements per step to count same-owner and enemy neighbors
# Index: (round_id, seed, step) -> list of (x, y, owner_id, alive)
settlement_index = {}
for round_id, seed, frames in replays:
    for frame in frames:
        step = frame["step"]
        key = (round_id[:8], seed, step)
        settlements = []
        for s in frame["settlements"]:
            if s["alive"]:
                settlements.append((s["x"], s["y"], s["owner_id"], s["has_port"], s["wealth"]))
        settlement_index[key] = settlements

def count_neighbors(row, max_dist=2):
    """Count same-owner and enemy settlements within Chebyshev distance."""
    key = (row["round_id"], row["seed"], row["step"])
    if key not in settlement_index:
        return 0, 0, 0, 0
    same = 0
    enemy = 0
    same_ports = 0
    enemy_wealth = 0.0
    for sx, sy, oid, has_port, w in settlement_index[key]:
        if sx == row["x"] and sy == row["y"]:
            continue
        d = max(abs(sx - row["x"]), abs(sy - row["y"]))
        if d <= max_dist:
            if oid == row["owner_id"]:
                same += 1
                if has_port:
                    same_ports += 1
            else:
                enemy += 1
                enemy_wealth += w
    return same, enemy, same_ports, enemy_wealth

# ================================================================
# Focus on positive d_wealth events
# ================================================================
pos = clean[clean["d_wealth"] > 1e-6].copy()
neg = clean[clean["d_wealth"] < -1e-6].copy()
zero = clean[clean["d_wealth"].abs() <= 1e-6].copy()

print(f"Positive d_wealth: {len(pos)}")
print(f"Negative d_wealth: {len(neg)}")
print(f"Zero d_wealth: {len(zero)}")

# Check d_wealth value distribution for positive cases
print(f"\nPositive d_wealth values:")
print(pos["d_wealth"].describe())
print(f"\nMost common positive d_wealth values (rounded to 3 decimals):")
print(pos["d_wealth"].round(3).value_counts().head(20))

print(f"\nNegative d_wealth values:")
print(neg["d_wealth"].describe())
print(f"\nMost common negative d_wealth values (rounded to 3 decimals):")
print(neg["d_wealth"].round(3).value_counts().head(20))

# ================================================================
# Check: is d_wealth symmetric? (gain ≈ loss like trade?)
# ================================================================
print(f"\n{'=' * 60}")
print("d_wealth symmetry check")
print("=" * 60)
print(f"Positive sum: {pos['d_wealth'].sum():.4f}")
print(f"Negative sum: {neg['d_wealth'].sum():.4f}")
print(f"Net:          {clean['d_wealth'].sum():.4f}")

# ================================================================
# Do positive d_wealth events cluster in time? (same step)
# ================================================================
print(f"\n{'=' * 60}")
print("Positive d_wealth by step")
print("=" * 60)

pos_by_step = pos.groupby("step").size()
print(f"Mean per step: {pos_by_step.mean():.1f}")
print(f"Std per step:  {pos_by_step.std():.1f}")

# ================================================================
# Port analysis: are ports more likely to gain wealth?
# ================================================================
print(f"\n{'=' * 60}")
print("Port vs non-port wealth changes")
print("=" * 60)

for port_val in [True, False]:
    sub = clean[clean["has_port"] == port_val]
    n_pos = (sub["d_wealth"] > 1e-6).sum()
    n_neg = (sub["d_wealth"] < -1e-6).sum()
    n_zero = (sub["d_wealth"].abs() <= 1e-6).sum()
    print(f"  has_port={port_val}: N={len(sub)}, "
          f"pos={n_pos} ({n_pos/len(sub)*100:.1f}%), "
          f"neg={n_neg} ({n_neg/len(sub)*100:.1f}%), "
          f"zero={n_zero} ({n_zero/len(sub)*100:.1f}%)")
    if n_pos > 0:
        print(f"    mean positive d_wealth: {sub[sub['d_wealth'] > 1e-6]['d_wealth'].mean():.6f}")

# ================================================================
# Compute enemy neighbor counts for a sample
# ================================================================
print(f"\n{'=' * 60}")
print("Neighbor analysis for positive d_wealth")
print("=" * 60)

# Sample for speed: all positive, and a random sample of zero
sample_pos = pos.copy()
sample_zero = zero.sample(min(5000, len(zero)), random_state=42).copy()
sample_neg = neg.sample(min(5000, len(neg)), random_state=42).copy()

for label, sample in [("positive", sample_pos), ("negative", sample_neg), ("zero", sample_zero)]:
    results = sample.apply(count_neighbors, axis=1, result_type="expand")
    sample["n_same_owner"] = results[0]
    sample["n_enemy"] = results[1]
    sample["n_same_ports"] = results[2]
    sample["enemy_wealth_nearby"] = results[3]
    print(f"\n  {label} d_wealth (N={len(sample)}):")
    print(f"    Same-owner neighbors (dist≤2): {sample['n_same_owner'].mean():.2f}")
    print(f"    Enemy neighbors (dist≤2):      {sample['n_enemy'].mean():.2f}")
    print(f"    Same-owner ports (dist≤2):     {sample['n_same_ports'].mean():.2f}")
    print(f"    Enemy wealth nearby:           {sample['enemy_wealth_nearby'].mean():.4f}")

# ================================================================
# Check if negative d_wealth = own settlement being raided nearby
# (wealth flows from raid victim to raider/nearby)
# ================================================================
print(f"\n{'=' * 60}")
print("Negative d_wealth and own-settlement raids")
print("=" * 60)

def has_own_raid_nearby(row, max_dist=2):
    key = (row["round_id"], row["seed"], row["step"])
    if key not in raid_index:
        return False
    for rx, ry, raid_owner in raid_index[key]:
        if max(abs(rx - row["x"]), abs(ry - row["y"])) <= max_dist:
            if raid_owner == row["owner_id"]:
                return True
    return False

def has_enemy_raid_nearby(row, max_dist=2):
    key = (row["round_id"], row["seed"], row["step"])
    if key not in raid_index:
        return False
    for rx, ry, raid_owner in raid_index[key]:
        if max(abs(rx - row["x"]), abs(ry - row["y"])) <= max_dist:
            if raid_owner != row["owner_id"]:
                return True
    return False

neg["own_raid_nearby"] = neg.apply(has_own_raid_nearby, axis=1)
neg["enemy_raid_nearby"] = neg.apply(has_enemy_raid_nearby, axis=1)
pos["own_raid_nearby"] = pos.apply(has_own_raid_nearby, axis=1)
pos["enemy_raid_nearby"] = pos.apply(has_enemy_raid_nearby, axis=1)

print(f"Negative d_wealth with own raid nearby: {neg['own_raid_nearby'].sum()} ({neg['own_raid_nearby'].mean()*100:.1f}%)")
print(f"Negative d_wealth with enemy raid nearby: {neg['enemy_raid_nearby'].sum()} ({neg['enemy_raid_nearby'].mean()*100:.1f}%)")
print(f"Positive d_wealth with own raid nearby: {pos['own_raid_nearby'].sum()} ({pos['own_raid_nearby'].mean()*100:.1f}%)")
print(f"Positive d_wealth with enemy raid nearby: {pos['enemy_raid_nearby'].sum()} ({pos['enemy_raid_nearby'].mean()*100:.1f}%)")

# ================================================================
# Within-group analysis: do wealth gains and losses net to zero
# within the same (round, seed, step)?
# ================================================================
print(f"\n{'=' * 60}")
print("Within-step wealth conservation")
print("=" * 60)

step_wealth = clean.groupby(["round_id", "seed", "step"])["d_wealth"].sum().reset_index()
print(f"Sum of d_wealth per (round, seed, step):")
print(f"  mean: {step_wealth['d_wealth'].mean():.6f}")
print(f"  std:  {step_wealth['d_wealth'].std():.6f}")
print(f"  median: {step_wealth['d_wealth'].median():.6f}")

# Include raided settlements too
step_wealth_all = alive[alive["alive_next"]].groupby(
    ["round_id", "seed", "step"])["d_wealth"].sum().reset_index()
print(f"\nSum of d_wealth per step (ALL alive settlements):")
print(f"  mean: {step_wealth_all['d_wealth'].mean():.6f}")
print(f"  std:  {step_wealth_all['d_wealth'].std():.6f}")

# ================================================================
# Plotting
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. d_wealth value distribution for positive cases
axes[0, 0].hist(pos["d_wealth"], bins=100, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("d_wealth")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title(f"Positive d_wealth values (N={len(pos)})")

# 2. d_wealth value distribution for negative cases
axes[0, 1].hist(neg["d_wealth"], bins=100, edgecolor="black", alpha=0.7, color="salmon")
axes[0, 1].set_xlabel("d_wealth")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title(f"Negative d_wealth values (N={len(neg)})")

# 3. P(positive d_wealth) by port status
port_rate = (clean[clean["has_port"]]["d_wealth"] > 1e-6).mean()
noport_rate = (clean[~clean["has_port"]]["d_wealth"] > 1e-6).mean()
axes[0, 2].bar(["Non-port", "Port"], [noport_rate, port_rate], alpha=0.7, color=["steelblue", "coral"])
axes[0, 2].set_ylabel("P(positive d_wealth)")
axes[0, 2].set_title("Wealth gain by port status")

# 4. Positive d_wealth: enemy raid vs own raid nearby
labels = ["Enemy\nraid nearby", "Own\nraid nearby", "Both", "Neither"]
enemy_only = pos["enemy_raid_nearby"] & ~pos["own_raid_nearby"]
own_only = pos["own_raid_nearby"] & ~pos["enemy_raid_nearby"]
both = pos["enemy_raid_nearby"] & pos["own_raid_nearby"]
neither = ~pos["enemy_raid_nearby"] & ~pos["own_raid_nearby"]
counts = [enemy_only.sum(), own_only.sum(), both.sum(), neither.sum()]
axes[1, 0].bar(labels, counts, alpha=0.7, color=["coral", "steelblue", "purple", "gray"])
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Positive d_wealth: what's nearby?")
for i, c in enumerate(counts):
    axes[1, 0].text(i, c + 100, f"{c}\n({c/len(pos)*100:.0f}%)", ha="center", fontsize=8)

# 5. Per-step net wealth change
axes[1, 1].hist(step_wealth_all["d_wealth"], bins=80, edgecolor="black", alpha=0.7)
axes[1, 1].set_xlabel("Sum(d_wealth) per step")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Net wealth change per step\n(all alive settlements)")
axes[1, 1].axvline(0, color="red", linestyle="--")

# 6. d_wealth by number of nearby ports (n_port from transition df)
for np_val in sorted(clean["n_port"].unique()):
    sub = clean[clean["n_port"] == np_val]
    if len(sub) > 100:
        rate = (sub["d_wealth"] > 1e-6).mean()
        axes[1, 2].bar(np_val, rate, alpha=0.7, width=0.8)
        axes[1, 2].text(np_val, rate + 0.003, f"n={len(sub)}", ha="center", fontsize=7)
axes[1, 2].set_xlabel("Adjacent ports (in grid)")
axes[1, 2].set_ylabel("P(positive d_wealth)")
axes[1, 2].set_title("Wealth gain rate by adjacent port count")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "wealth_analysis_v2.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "wealth_analysis_v2.png"), dpi=150)
print(f"\nSaved: {OUT}/wealth_analysis_v2.pdf")
