"""
Investigate wealth mechanics:
  H1: Wealth stays constant almost always
  H2: Wealth only changes positively when a raid happens within Chebyshev distance 2 (5x5 box)
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

# ================================================================
# H1: Does wealth stay constant almost always?
# ================================================================
print("=" * 60)
print("H1: Does wealth stay constant almost always?")
print("=" * 60)

# Overall d_wealth distribution
print(f"\nAll alive transitions: {len(alive)}")
print(f"d_wealth stats:")
print(f"  mean:   {alive['d_wealth'].mean():.6f}")
print(f"  std:    {alive['d_wealth'].std():.6f}")
print(f"  median: {alive['d_wealth'].median():.6f}")

# How many have d_wealth ≈ 0?
for thresh in [1e-6, 1e-4, 1e-3, 0.005, 0.01]:
    n_zero = (alive["d_wealth"].abs() < thresh).sum()
    print(f"  |d_wealth| < {thresh}: {n_zero} ({n_zero/len(alive)*100:.1f}%)")

# Non-raided, same-owner: isolate pure wealth dynamics
clean = alive[(alive["d_defense"] >= -0.001) & alive["owner_same"]].copy()
print(f"\nNon-raided, same-owner: {len(clean)}")
print(f"  d_wealth mean:   {clean['d_wealth'].mean():.6f}")
print(f"  d_wealth std:    {clean['d_wealth'].std():.6f}")
n_exact_zero = (clean["d_wealth"] == 0.0).sum()
print(f"  d_wealth exactly 0: {n_exact_zero} ({n_exact_zero/len(clean)*100:.1f}%)")

for thresh in [1e-6, 1e-4, 1e-3, 0.005]:
    n_z = (clean["d_wealth"].abs() < thresh).sum()
    print(f"  |d_wealth| < {thresh}: {n_z} ({n_z/len(clean)*100:.1f}%)")

# Positive vs negative vs zero
n_pos = (clean["d_wealth"] > 1e-6).sum()
n_neg = (clean["d_wealth"] < -1e-6).sum()
n_zero = (clean["d_wealth"].abs() <= 1e-6).sum()
print(f"\n  Positive: {n_pos} ({n_pos/len(clean)*100:.1f}%)")
print(f"  Negative: {n_neg} ({n_neg/len(clean)*100:.1f}%)")
print(f"  Zero:     {n_zero} ({n_zero/len(clean)*100:.1f}%)")

# ================================================================
# H2: Wealth increases only near raids (5x5 = Chebyshev dist ≤ 2)
# ================================================================
print(f"\n{'=' * 60}")
print("H2: Wealth changes only near raids?")
print("=" * 60)

# Need to identify raids per (round, seed, step) and check proximity
# Build per-step raid location index
print("Building per-step raid index...")
raided = alive[alive["d_defense"] < -0.01].copy()
print(f"Total raid events: {len(raided)}")

# For each transition, check if there's a raid within Chebyshev dist ≤ 2
# Group raids by (round_id, seed, step)
raid_index = {}
for _, r in raided.iterrows():
    key = (r["round_id"], r["seed"], r["step"])
    if key not in raid_index:
        raid_index[key] = []
    raid_index[key].append((r["x"], r["y"], r["owner_id"]))

def has_nearby_raid(row, max_dist=2):
    """Check if any raid happened within Chebyshev distance max_dist."""
    key = (row["round_id"], row["seed"], row["step"])
    if key not in raid_index:
        return False
    for rx, ry, _ in raid_index[key]:
        if max(abs(rx - row["x"]), abs(ry - row["y"])) <= max_dist:
            return True
    return False

def has_nearby_enemy_raid(row, max_dist=2):
    """Check if a raid on an ENEMY settlement happened within Chebyshev distance max_dist."""
    key = (row["round_id"], row["seed"], row["step"])
    if key not in raid_index:
        return False
    for rx, ry, raid_owner in raid_index[key]:
        if max(abs(rx - row["x"]), abs(ry - row["y"])) <= max_dist:
            if raid_owner != row["owner_id"]:
                return True
    return False

def has_nearby_own_raid(row, max_dist=2):
    """Check if a raid on OWN settlement happened within Chebyshev distance max_dist."""
    key = (row["round_id"], row["seed"], row["step"])
    if key not in raid_index:
        return False
    for rx, ry, raid_owner in raid_index[key]:
        if max(abs(rx - row["x"]), abs(ry - row["y"])) <= max_dist:
            if raid_owner == row["owner_id"]:
                return True
    return False

# Apply to non-raided same-owner settlements (the "clean" set)
print("Checking proximity to raids for clean transitions...")
clean["raid_nearby"] = clean.apply(has_nearby_raid, axis=1)
clean["enemy_raid_nearby"] = clean.apply(has_nearby_enemy_raid, axis=1)
clean["own_raid_nearby"] = clean.apply(has_nearby_own_raid, axis=1)

print(f"\nClean settlements with raid within dist 2: {clean['raid_nearby'].sum()} ({clean['raid_nearby'].mean()*100:.1f}%)")
print(f"  - enemy raided nearby: {clean['enemy_raid_nearby'].sum()} ({clean['enemy_raid_nearby'].mean()*100:.1f}%)")
print(f"  - own raided nearby:   {clean['own_raid_nearby'].sum()} ({clean['own_raid_nearby'].mean()*100:.1f}%)")

# Compare d_wealth for nearby-raid vs no-nearby-raid
for label, col in [("any raid nearby", "raid_nearby"),
                    ("enemy raid nearby", "enemy_raid_nearby"),
                    ("own raid nearby", "own_raid_nearby")]:
    near = clean[clean[col]]
    far = clean[~clean[col]]
    print(f"\n  {label}:")
    print(f"    WITH:    n={len(near)}, d_wealth mean={near['d_wealth'].mean():.6f}, "
          f"positive={( near['d_wealth'] > 1e-6).sum()}/{len(near)} ({(near['d_wealth'] > 1e-6).mean()*100:.1f}%)")
    print(f"    WITHOUT: n={len(far)}, d_wealth mean={far['d_wealth'].mean():.6f}, "
          f"positive={( far['d_wealth'] > 1e-6).sum()}/{len(far)} ({(far['d_wealth'] > 1e-6).mean()*100:.1f}%)")

# Key test: are ALL positive d_wealth events near raids?
pos_wealth = clean[clean["d_wealth"] > 1e-6]
print(f"\n--- Positive d_wealth events (non-raided, same-owner) ---")
print(f"Total: {len(pos_wealth)}")
print(f"  With any raid nearby (dist≤2): {pos_wealth['raid_nearby'].sum()} ({pos_wealth['raid_nearby'].mean()*100:.1f}%)")
print(f"  With enemy raid nearby: {pos_wealth['enemy_raid_nearby'].sum()} ({pos_wealth['enemy_raid_nearby'].mean()*100:.1f}%)")
print(f"  With own raid nearby: {pos_wealth['own_raid_nearby'].sum()} ({pos_wealth['own_raid_nearby'].mean()*100:.1f}%)")

# Check the positive-wealth cases WITHOUT nearby raids
pos_no_raid = pos_wealth[~pos_wealth["raid_nearby"]]
if len(pos_no_raid) > 0:
    print(f"\n  Positive d_wealth WITHOUT nearby raid: {len(pos_no_raid)}")
    print(f"    d_wealth stats: mean={pos_no_raid['d_wealth'].mean():.6f}, max={pos_no_raid['d_wealth'].max():.6f}")
    print(f"    has_port distribution: {pos_no_raid['has_port'].value_counts().to_dict()}")
    print(f"    wealth before: mean={pos_no_raid['wealth'].mean():.4f}")
else:
    print(f"\n  ALL positive d_wealth events have a raid nearby!")

# Also check: negative d_wealth without nearby raids
neg_wealth = clean[clean["d_wealth"] < -1e-6]
print(f"\n--- Negative d_wealth events (non-raided, same-owner) ---")
print(f"Total: {len(neg_wealth)}")
print(f"  With any raid nearby: {neg_wealth['raid_nearby'].sum()} ({neg_wealth['raid_nearby'].mean()*100:.1f}%)")

# ================================================================
# Check at larger radii
# ================================================================
print(f"\n{'=' * 60}")
print("Checking different radii for the raid proximity hypothesis")
print("=" * 60)

for max_d in [1, 2, 3, 4, 5]:
    def nearby_at_dist(row, md=max_d):
        key = (row["round_id"], row["seed"], row["step"])
        if key not in raid_index:
            return False
        for rx, ry, _ in raid_index[key]:
            if max(abs(rx - row["x"]), abs(ry - row["y"])) <= md:
                return True
        return False

    col = f"raid_d{max_d}"
    clean[col] = clean.apply(nearby_at_dist, axis=1)
    pos_with = pos_wealth.index.isin(clean[clean[col]].index)
    # Recompute for pos_wealth
    pos_wealth_col = pos_wealth.apply(nearby_at_dist, axis=1)
    coverage = pos_wealth_col.sum() / len(pos_wealth) * 100 if len(pos_wealth) > 0 else 0
    print(f"  Chebyshev dist ≤ {max_d}: {coverage:.1f}% of positive d_wealth explained")

# ================================================================
# Plotting
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. d_wealth histogram (all alive)
axes[0, 0].hist(alive["d_wealth"].clip(-0.1, 0.1), bins=200, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("d_wealth")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title(f"d_wealth distribution (all alive, N={len(alive)})")
axes[0, 0].axvline(0, color="red", linestyle="--")

# 2. d_wealth histogram zoomed (non-raided, same-owner)
axes[0, 1].hist(clean["d_wealth"].clip(-0.05, 0.05), bins=200, edgecolor="black", alpha=0.7)
axes[0, 1].set_xlabel("d_wealth")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title(f"d_wealth (non-raided, same-owner, N={len(clean)})")
axes[0, 1].axvline(0, color="red", linestyle="--")

# 3. d_wealth: near-raid vs no-raid
near = clean[clean["raid_nearby"]]
far = clean[~clean["raid_nearby"]]
axes[0, 2].hist(far["d_wealth"].clip(-0.05, 0.05), bins=100, alpha=0.5, density=True, label=f"No raid nearby (N={len(far)})")
axes[0, 2].hist(near["d_wealth"].clip(-0.05, 0.05), bins=100, alpha=0.5, density=True, label=f"Raid nearby (N={len(near)})")
axes[0, 2].set_xlabel("d_wealth")
axes[0, 2].set_ylabel("Density")
axes[0, 2].set_title("d_wealth: near raid vs no raid")
axes[0, 2].legend(fontsize=8)

# 4. Positive d_wealth rate by proximity to raids
near_rate = (near["d_wealth"] > 1e-6).mean()
far_rate = (far["d_wealth"] > 1e-6).mean()
axes[1, 0].bar(["No raid\nnearby", "Raid\nnearby"], [far_rate, near_rate], alpha=0.7, color=["steelblue", "coral"])
axes[1, 0].set_ylabel("P(d_wealth > 0)")
axes[1, 0].set_title("P(positive wealth change)")
for i, v in enumerate([far_rate, near_rate]):
    axes[1, 0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

# 5. d_wealth vs wealth for settlements near raids
axes[1, 1].scatter(near["wealth"], near["d_wealth"], alpha=0.15, s=3, color="coral", label="Raid nearby")
axes[1, 1].scatter(far["wealth"].sample(min(5000, len(far)), random_state=42),
                   far.loc[far["wealth"].sample(min(5000, len(far)), random_state=42).index, "d_wealth"],
                   alpha=0.1, s=3, color="steelblue", label="No raid nearby")
axes[1, 1].set_xlabel("Wealth (before)")
axes[1, 1].set_ylabel("d_wealth")
axes[1, 1].set_title("d_wealth vs wealth")
axes[1, 1].legend(fontsize=8)

# 6. Positive d_wealth coverage by raid radius
radii = [1, 2, 3, 4, 5]
coverages = []
for max_d in radii:
    def nearby_at_dist(row, md=max_d):
        key = (row["round_id"], row["seed"], row["step"])
        if key not in raid_index:
            return False
        for rx, ry, _ in raid_index[key]:
            if max(abs(rx - row["x"]), abs(ry - row["y"])) <= md:
                return True
        return False
    cov = pos_wealth.apply(nearby_at_dist, axis=1).mean() * 100 if len(pos_wealth) > 0 else 0
    coverages.append(cov)

axes[1, 2].bar(radii, coverages, alpha=0.7)
axes[1, 2].set_xlabel("Max Chebyshev distance to raid")
axes[1, 2].set_ylabel("% of positive d_wealth explained")
axes[1, 2].set_title("Raid proximity explains positive wealth")
for i, (r, c) in enumerate(zip(radii, coverages)):
    axes[1, 2].text(r, c + 1, f"{c:.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "wealth_analysis.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "wealth_analysis.png"), dpi=150)
print(f"\nSaved: {OUT}/wealth_analysis.pdf")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'=' * 60}")
print("WEALTH ANALYSIS SUMMARY")
print("=" * 60)
n_zero_clean = (clean["d_wealth"].abs() < 1e-6).sum()
print(f"""
H1: Wealth stays constant almost always
  Non-raided, same-owner transitions: {len(clean)}
  d_wealth exactly 0: {n_exact_zero} ({n_exact_zero/len(clean)*100:.1f}%)
  |d_wealth| < 0.001: {(clean['d_wealth'].abs() < 0.001).sum()} ({(clean['d_wealth'].abs() < 0.001).mean()*100:.1f}%)
  Positive changes: {n_pos} ({n_pos/len(clean)*100:.1f}%)
  Negative changes: {n_neg} ({n_neg/len(clean)*100:.1f}%)

H2: Positive wealth only near raids (Chebyshev dist ≤ 2)
  Positive d_wealth with raid nearby: {pos_wealth['raid_nearby'].sum()}/{len(pos_wealth)} ({pos_wealth['raid_nearby'].mean()*100:.1f}%)
  Positive d_wealth WITHOUT raid nearby: {len(pos_no_raid) if len(pos_no_raid) > 0 else 0}
""")
