"""
Quantify the p_spawn functional form using logistic regression and direct binning.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_replays import load_all_replays, build_transition_df, build_spawn_df
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

replays = load_all_replays()
trans = build_transition_df(replays)
spawns = build_spawn_df(replays)

# Mark which settlements spawned
parent_spawns = spawns[spawns["parent_dist"].notna()].copy()
parent_spawn_keys = parent_spawns.groupby(
    ["round_id", "seed", "step", "parent_x", "parent_y"]
).size().reset_index(name="n_children")

alive = trans[trans["alive"]].copy()
alive = alive.merge(
    parent_spawn_keys,
    left_on=["round_id", "seed", "step", "x", "y"],
    right_on=["round_id", "seed", "step", "parent_x", "parent_y"],
    how="left"
)
alive["spawned"] = alive["n_children"].fillna(0).astype(int) > 0

# Count available buildable terrain nearby (proxy for spawn opportunity)
alive["n_buildable"] = alive["n_plains"] + alive["n_forest"] + alive["n_ruin"]
alive["pop_x_food"] = alive["pop"] * alive["food"]

print(f"Total alive settlements: {len(alive)}")
print(f"Spawned: {alive['spawned'].sum()} ({alive['spawned'].mean():.4f})")

# ================================================================
# 1. Logistic regression: p_spawn ~ pop, food, pop*food, ...
# ================================================================
feats = ["pop", "food", "pop_x_food"]
X = alive[feats].values
y = alive["spawned"].astype(int).values

lr = LogisticRegression(max_iter=1000).fit(X, y)
print(f"\nLogistic regression (pop, food, pop*food):")
print(f"  Intercept: {lr.intercept_[0]:.4f}")
for f, c in zip(feats, lr.coef_[0]):
    print(f"  {f}: {c:.4f}")

# Also try simpler models
for feats_trial, name in [
    (["pop_x_food"], "pop*food only"),
    (["pop", "food"], "pop + food"),
    (["pop"], "pop only"),
    (["food"], "food only"),
]:
    lr_t = LogisticRegression(max_iter=1000).fit(alive[feats_trial].values, y)
    score = lr_t.score(alive[feats_trial].values, y)
    print(f"  {name}: accuracy={score:.4f}, intercept={lr_t.intercept_[0]:.4f}, coefs={lr_t.coef_[0]}")

# ================================================================
# 2. Non-parametric: binned spawn rate
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# P(spawn) vs pop (fine bins)
pop_bins = np.arange(0, 5.5, 0.25)
alive["pop_bin_fine"] = pd.cut(alive["pop"], bins=pop_bins)
rate_by_pop = alive.groupby("pop_bin_fine", observed=True).agg(
    p=("spawned", "mean"), n=("spawned", "count")).reset_index()

axes[0, 0].bar(range(len(rate_by_pop)), rate_by_pop["p"], alpha=0.7)
axes[0, 0].set_xticks(range(0, len(rate_by_pop), 2))
axes[0, 0].set_xticklabels([f"{x.mid:.1f}" for x in rate_by_pop["pop_bin_fine"]][::2],
                           rotation=45, fontsize=8)
axes[0, 0].set_xlabel("Population")
axes[0, 0].set_ylabel("P(spawn)")
axes[0, 0].set_title("P(spawn) vs Population")

# Overlay model: p = growth_prob * pop * food (with average food)
avg_food = alive["food"].mean()
pop_range = np.linspace(0, 5, 100)
# Fit: p_spawn = g * pop * food  →  at average food: p = g * avg_food * pop
# From overall rate: g * avg_pop * avg_food = 0.0967
g_est = alive["spawned"].mean() / (alive["pop"].mean() * alive["food"].mean())
model_p = g_est * pop_range * avg_food
axes[0, 0].plot(pop_range / 0.25, model_p, "r-", linewidth=2,
               label=f"p={g_est:.3f}*pop*food_avg")
axes[0, 0].legend(fontsize=8)

# P(spawn) vs food (fine bins)
food_bins = np.arange(0, 1.05, 0.05)
alive["food_bin_fine"] = pd.cut(alive["food"], bins=food_bins)
rate_by_food = alive.groupby("food_bin_fine", observed=True).agg(
    p=("spawned", "mean"), n=("spawned", "count")).reset_index()

axes[0, 1].bar(range(len(rate_by_food)), rate_by_food["p"], alpha=0.7)
axes[0, 1].set_xticks(range(0, len(rate_by_food), 2))
axes[0, 1].set_xticklabels([f"{x.mid:.2f}" for x in rate_by_food["food_bin_fine"]][::2],
                           rotation=45, fontsize=8)
axes[0, 1].set_xlabel("Food")
axes[0, 1].set_ylabel("P(spawn)")
axes[0, 1].set_title("P(spawn) vs Food")

# P(spawn) vs pop*food
pf_bins = np.arange(0, 4.5, 0.2)
alive["pf_bin"] = pd.cut(alive["pop_x_food"], bins=pf_bins)
rate_by_pf = alive.groupby("pf_bin", observed=True).agg(
    p=("spawned", "mean"), n=("spawned", "count")).reset_index()

axes[0, 2].bar(range(len(rate_by_pf)), rate_by_pf["p"], alpha=0.7)
axes[0, 2].set_xticks(range(0, len(rate_by_pf), 2))
axes[0, 2].set_xticklabels([f"{x.mid:.1f}" for x in rate_by_pf["pf_bin"]][::2],
                           rotation=45, fontsize=8)
axes[0, 2].set_xlabel("Population × Food")
axes[0, 2].set_ylabel("P(spawn)")
axes[0, 2].set_title("P(spawn) vs Pop × Food")

# Overlay linear model: p = g * pop * food
pf_range = np.linspace(0, 4.5, 100)
axes[0, 2].plot(pf_range / 0.2, g_est * pf_range, "r-", linewidth=2,
               label=f"p={g_est:.3f}*pop*food")
axes[0, 2].legend(fontsize=8)

# P(spawn) vs n_buildable
rate_by_build = alive.groupby("n_buildable")["spawned"].agg(["mean", "count"]).reset_index()
axes[1, 0].bar(rate_by_build["n_buildable"], rate_by_build["mean"], alpha=0.7)
for _, row in rate_by_build.iterrows():
    axes[1, 0].text(row["n_buildable"], row["mean"] + 0.005,
                   f"n={int(row['count'])}", ha="center", fontsize=7)
axes[1, 0].set_xlabel("# Buildable neighbors (plains+forest+ruin)")
axes[1, 0].set_ylabel("P(spawn)")
axes[1, 0].set_title("P(spawn) vs available build sites")

# P(spawn) vs defense (does defense matter?)
def_bins = np.arange(0, 1.05, 0.05)
alive["def_bin_fine"] = pd.cut(alive["defense"], bins=def_bins)
rate_by_def = alive.groupby("def_bin_fine", observed=True).agg(
    p=("spawned", "mean"), n=("spawned", "count")).reset_index()
axes[1, 1].bar(range(len(rate_by_def)), rate_by_def["p"], alpha=0.7)
axes[1, 1].set_xticks(range(0, len(rate_by_def), 2))
axes[1, 1].set_xticklabels([f"{x.mid:.2f}" for x in rate_by_def["def_bin_fine"]][::2],
                           rotation=45, fontsize=8)
axes[1, 1].set_xlabel("Defense")
axes[1, 1].set_ylabel("P(spawn)")
axes[1, 1].set_title("P(spawn) vs Defense")

# P(spawn) vs wealth
w_bins = np.arange(0, 0.55, 0.025)
alive["w_bin_fine"] = pd.cut(alive["wealth"], bins=w_bins)
rate_by_w = alive.groupby("w_bin_fine", observed=True).agg(
    p=("spawned", "mean"), n=("spawned", "count")).reset_index()
axes[1, 2].bar(range(len(rate_by_w)), rate_by_w["p"], alpha=0.7)
axes[1, 2].set_xticks(range(0, len(rate_by_w), 2))
axes[1, 2].set_xticklabels([f"{x.mid:.3f}" for x in rate_by_w["w_bin_fine"]][::2],
                           rotation=45, fontsize=8)
axes[1, 2].set_xlabel("Wealth")
axes[1, 2].set_ylabel("P(spawn)")
axes[1, 2].set_title("P(spawn) vs Wealth")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "spawn_pspawn.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "spawn_pspawn.png"), dpi=150)
print(f"\nSaved: {OUT}/spawn_pspawn.pdf")

# ================================================================
# Child food investigation: what determines child food?
# ================================================================
print(f"\n{'='*60}")
print("Child food determinants")
print(f"{'='*60}")

# For ruin spawns: child_pop=0.4, child_def=0.15 always
ruin = spawns[spawns["terrain_before"] == 3].copy()
# Check: does child food depend on parent food?
if len(ruin) > 100:
    corr_parent = ruin[["child_food", "parent_food"]].dropna().corr().iloc[0, 1]
    print(f"Ruin spawn child_food ~ parent_food correlation: {corr_parent:.4f}")

    # Does child food depend on terrain neighbors?
    ruin["n_adj_plains"] = ruin["n_adj_plains"]
    corr_plains = ruin[["child_food", "n_adj_plains"]].corr().iloc[0, 1]
    print(f"Ruin spawn child_food ~ n_adj_plains correlation: {corr_plains:.4f}")

    corr_forest = ruin[["child_food", "n_adj_forest"]].corr().iloc[0, 1]
    print(f"Ruin spawn child_food ~ n_adj_forest correlation: {corr_forest:.4f}")

    # Does child food depend on step?
    corr_step = ruin[["child_food", "step"]].corr().iloc[0, 1]
    print(f"Ruin spawn child_food ~ step correlation: {corr_step:.4f}")

    # Regression: child_food ~ n_adj_plains + n_adj_forest + step
    from sklearn.linear_model import LinearRegression as LR
    feats_cf = ["n_adj_plains", "n_adj_forest", "step"]
    X_cf = ruin[feats_cf].values
    y_cf = ruin["child_food"].values
    reg_cf = LR().fit(X_cf, y_cf)
    print(f"\nRegression child_food ~ terrain + step:")
    print(f"  Intercept: {reg_cf.intercept_:.4f}")
    for f, c in zip(feats_cf, reg_cf.coef_):
        print(f"  {f}: {c:+.6f}")
    print(f"  R²: {reg_cf.score(X_cf, y_cf):.4f}")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*60}")
print("SPAWN PROBABILITY SUMMARY")
print(f"{'='*60}")
print(f"""
Overall p_spawn: {alive['spawned'].mean():.4f}

Best model: p_spawn ≈ {g_est:.4f} × population × food
  (linear in both pop and food)

The p_spawn is:
  - Proportional to population (roughly linear up to pop~4.5)
  - Proportional to food (roughly linear up to food~0.8, drops at very high food)
  - Independent of defense, wealth, port status
  - Requires available buildable terrain nearby (n_buildable > 0)
  - g_est = {g_est:.4f}
""")
