"""
Q5-Q8: Raid mechanics analysis.

Detect raids by defense drops (d_defense < -0.05 to exclude natural recovery).
Pair with nearest enemy as attacker.
Separate successful raids (ownership change or death) from unsuccessful ones.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_replays import load_all_replays, build_raid_df

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

replays = load_all_replays()
raids = build_raid_df(replays)

print(f"Total raid events (defense drops): {len(raids)}")
print(f"Successful (ownership changed): {raids['owner_changed'].sum()}")
print(f"Unsuccessful: {(~raids['owner_changed']).sum()}")

# Separate successful and unsuccessful
unsuccessful = raids[~raids["owner_changed"]].copy()
successful = raids[raids["owner_changed"]].copy()

print(f"\n{'='*60}")
print("Q5: Defender changes on UNSUCCESSFUL raid")
print(f"{'='*60}")
print(f"N = {len(unsuccessful)}")
for col in ["d_def_pop", "d_def_food", "d_def_wealth", "d_def_defense"]:
    print(f"  {col}: mean={unsuccessful[col].mean():.4f}, std={unsuccessful[col].std():.4f}, "
          f"median={unsuccessful[col].median():.4f}")

# Defender pop ratio (pop_next / pop)
unsuccessful["def_pop_ratio"] = unsuccessful["def_pop_next"] / unsuccessful["def_pop"]
print(f"\n  Pop ratio (next/current): mean={unsuccessful['def_pop_ratio'].mean():.4f}, "
      f"std={unsuccessful['def_pop_ratio'].std():.4f}")
print(f"  Pop ratio unique values (first 20): {sorted(unsuccessful['def_pop_ratio'].dropna().round(4).unique())[:20]}")

# Defense delta distribution
print(f"\n  Defense delta histogram:")
def_d_vals = unsuccessful["d_def_defense"].round(3)
print(def_d_vals.value_counts().head(20))

print(f"\n{'='*60}")
print("Q6: Defender changes on SUCCESSFUL raid")
print(f"{'='*60}")
print(f"N = {len(successful)}")
for col in ["d_def_pop", "d_def_food", "d_def_wealth", "d_def_defense"]:
    print(f"  {col}: mean={successful[col].mean():.4f}, std={successful[col].std():.4f}, "
          f"median={successful[col].median():.4f}")

print(f"\n{'='*60}")
print("Q7: Attacker changes on UNSUCCESSFUL raid")
print(f"{'='*60}")
has_atk = unsuccessful[unsuccessful["atk_pop"].notna()].copy()
print(f"N = {len(has_atk)} (with identified attacker)")
for col in ["d_atk_pop", "d_atk_food", "d_atk_wealth", "d_atk_defense"]:
    print(f"  {col}: mean={has_atk[col].mean():.4f}, std={has_atk[col].std():.4f}")

# Wealth stolen: attacker wealth gain
print(f"\n  Attacker wealth gain: mean={has_atk['d_atk_wealth'].mean():.4f}")
# vs defender wealth loss
print(f"  Defender wealth loss: mean={has_atk['d_def_wealth'].mean():.4f}")

# Check if wealth gain = -wealth loss (zero sum?)
has_atk["wealth_sum"] = has_atk["d_atk_wealth"] + has_atk["d_def_wealth"]
print(f"  Sum (atk_gain + def_loss): mean={has_atk['wealth_sum'].mean():.4f}")

print(f"\n{'='*60}")
print("Q8: Attacker changes on SUCCESSFUL raid")
print(f"{'='*60}")
has_atk_succ = successful[successful["atk_pop"].notna()].copy()
print(f"N = {len(has_atk_succ)} (with identified attacker)")
for col in ["d_atk_pop", "d_atk_food", "d_atk_wealth", "d_atk_defense"]:
    print(f"  {col}: mean={has_atk_succ[col].mean():.4f}, std={has_atk_succ[col].std():.4f}")

# --- Analysis: What determines raid outcome? ---
print(f"\n{'='*60}")
print("Raid outcome determinants")
print(f"{'='*60}")
print(f"\nDefender defense at time of raid:")
print(f"  Unsuccessful: mean={unsuccessful['def_defense'].mean():.4f}")
print(f"  Successful: mean={successful['def_defense'].mean():.4f}")

# Defense after raid for unsuccessful
print(f"\nDefender defense AFTER raid:")
print(f"  Unsuccessful: mean={unsuccessful['def_defense_next'].mean():.4f}")
print(f"  Successful: mean={successful['def_defense_next'].mean():.4f}")

# Number of raids (multiple raids can happen)
raid_counts = raids.groupby(["round_id", "seed", "step", "def_x", "def_y"]).size()
print(f"\nRaids per defender per step: {raid_counts.value_counts().to_dict()}")

# --- PLOTS ---
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Row 1: Defender changes (unsuccessful)
axes[0, 0].hist(unsuccessful["d_def_defense"], bins=50, alpha=0.7, edgecolor="black")
axes[0, 0].set_xlabel("Defense delta")
axes[0, 0].set_title("Unsuccessful raid:\nDefender defense delta")

axes[0, 1].hist(unsuccessful["d_def_pop"].clip(-2, 2), bins=50, alpha=0.7, edgecolor="black")
axes[0, 1].set_xlabel("Population delta")
axes[0, 1].set_title("Unsuccessful raid:\nDefender pop delta")

axes[0, 2].hist(unsuccessful["d_def_food"].clip(-0.5, 0.5), bins=50, alpha=0.7, edgecolor="black")
axes[0, 2].set_xlabel("Food delta")
axes[0, 2].set_title("Unsuccessful raid:\nDefender food delta")

axes[0, 3].hist(unsuccessful["d_def_wealth"].clip(-0.5, 0.5), bins=50, alpha=0.7, edgecolor="black")
axes[0, 3].set_xlabel("Wealth delta")
axes[0, 3].set_title("Unsuccessful raid:\nDefender wealth delta")

# Row 2: Defender changes (successful)
axes[1, 0].hist(successful["d_def_defense"], bins=50, alpha=0.7, edgecolor="black", color="red")
axes[1, 0].set_xlabel("Defense delta")
axes[1, 0].set_title("Successful raid:\nDefender defense delta")

axes[1, 1].hist(successful["d_def_pop"].clip(-2, 2), bins=50, alpha=0.7, edgecolor="black", color="red")
axes[1, 1].set_xlabel("Population delta")
axes[1, 1].set_title("Successful raid:\nDefender pop delta")

axes[1, 2].hist(successful["d_def_food"].clip(-0.5, 0.5), bins=50, alpha=0.7, edgecolor="black", color="red")
axes[1, 2].set_xlabel("Food delta")
axes[1, 2].set_title("Successful raid:\nDefender food delta")

axes[1, 3].hist(successful["d_def_wealth"].clip(-0.5, 0.5), bins=50, alpha=0.7, edgecolor="black", color="red")
axes[1, 3].set_xlabel("Wealth delta")
axes[1, 3].set_title("Successful raid:\nDefender wealth delta")

# Row 3: Attacker changes and scatter
has_atk_all = raids[raids["atk_pop"].notna()].copy()
axes[2, 0].scatter(has_atk_all["def_defense"], has_atk_all["d_def_defense"],
                   c=has_atk_all["owner_changed"].astype(int), cmap="RdYlGn_r", alpha=0.3, s=5)
axes[2, 0].set_xlabel("Defender defense (before)")
axes[2, 0].set_ylabel("Defense delta")
axes[2, 0].set_title("Defense delta vs initial defense\n(green=unchanged, red=taken)")

# Pop survival ratio
has_atk_all["def_pop_ratio"] = has_atk_all["def_pop_next"] / has_atk_all["def_pop"]
axes[2, 1].hist(has_atk_all["def_pop_ratio"].clip(0, 1.5), bins=50, alpha=0.7, edgecolor="black")
axes[2, 1].set_xlabel("Pop ratio (next/current)")
axes[2, 1].set_title("Defender pop survival ratio")

# Wealth transfer
axes[2, 2].scatter(has_atk_all["d_def_wealth"], has_atk_all["d_atk_wealth"],
                   alpha=0.3, s=5)
axes[2, 2].set_xlabel("Defender wealth delta")
axes[2, 2].set_ylabel("Attacker wealth delta")
axes[2, 2].set_title("Wealth transfer (atk gain vs def loss)")
axes[2, 2].axhline(0, color="gray", linestyle="--")
axes[2, 2].axvline(0, color="gray", linestyle="--")

# Attacker distance
axes[2, 3].hist(has_atk_all["atk_dist"].clip(0, 10), bins=50, alpha=0.7, edgecolor="black")
axes[2, 3].set_xlabel("Euclidean distance (attacker → defender)")
axes[2, 3].set_title("Attacker-defender distance")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "raid_mechanics.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "raid_mechanics.png"), dpi=150)
print(f"\nSaved: {OUT}/raid_mechanics.pdf")

# --- Plot 2: Defense delta decomposition ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Defense delta is discrete? Check
defense_deltas = raids["d_def_defense"].round(4)
print(f"\nMost common defense deltas:")
print(defense_deltas.value_counts().head(20))

# Plot defense delta vs defender defense
axes[0, 0].scatter(raids["def_defense"], raids["d_def_defense"], alpha=0.2, s=5)
axes[0, 0].set_xlabel("Defender defense (before)")
axes[0, 0].set_ylabel("Defense delta")
axes[0, 0].set_title("Defense delta vs current defense")

# Plot pop survival vs defense
axes[0, 1].scatter(raids["def_defense"], raids["d_def_pop"] / raids["def_pop"].clip(0.001),
                   alpha=0.2, s=5, c=raids["owner_changed"].astype(int), cmap="RdYlGn_r")
axes[0, 1].set_xlabel("Defender defense (before)")
axes[0, 1].set_ylabel("Pop delta / Pop")
axes[0, 1].set_title("Pop loss fraction vs defense")

# Wealth stolen vs defender wealth
axes[1, 0].scatter(raids["def_wealth"], raids["d_def_wealth"], alpha=0.2, s=5)
axes[1, 0].set_xlabel("Defender wealth (before)")
axes[1, 0].set_ylabel("Defender wealth delta")
axes[1, 0].set_title("Wealth loss vs defender wealth")

# Success threshold
axes[1, 1].scatter(raids["def_defense_next"], raids["owner_changed"].astype(int), alpha=0.1, s=5)
axes[1, 1].set_xlabel("Defender defense after raid")
axes[1, 1].set_ylabel("Ownership changed (1=yes)")
axes[1, 1].set_title("Takeover probability vs post-raid defense")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "raid_details.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "raid_details.png"), dpi=150)
print(f"Saved: {OUT}/raid_details.pdf")
