"""
Q1: How large is the impact from winter on food?

Strategy:
- Filter to non-raid transitions (d_defense >= 0) and alive settlements
- Fit a linear food production model: food_next = f(food, pop, terrain)
- The residual reveals the weather/winter component
- Weather is constant within each (round, seed, step) group
- Extract the per-step weather component and measure its magnitude
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load_replays import load_all_replays, build_transition_df
from sklearn.linear_model import LinearRegression

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "plots")
os.makedirs(OUT, exist_ok=True)

# Load data
replays = load_all_replays()
df = build_transition_df(replays)

# Filter: alive, no raids (d_defense >= 0), settlement still alive next step
mask = (df["alive"]) & (df["alive_next"]) & (df["d_defense"] >= -0.001)
# Also exclude settlements that changed owner (takeover)
mask &= (df["owner_id"] == df["owner_id_next"])
clean = df[mask].copy()
print(f"Clean (no-raid, alive) transitions: {len(clean)}")

# --- Step 1: Fit a linear food model ---
# food_next - food = intercept + b1*pop + b2*food*(1-food) + b3*n_plains + b4*n_forest + b5*n_mountain + b6*n_settlement_adj + weather
# But weather is per (round, seed, step), so we use fixed effects

# First, simple OLS ignoring weather
X_cols = ["pop", "n_plains", "n_forest", "n_mountain"]
clean["food_x_1mfood"] = clean["food"] * (1 - clean["food"])
clean["n_settl_adj"] = clean["n_settlement"] + clean["n_port"] + clean["n_ruin"]

features = ["pop", "food_x_1mfood", "n_plains", "n_forest", "n_mountain", "n_settl_adj"]
X = clean[features].values
y = clean["d_food"].values

reg = LinearRegression().fit(X, y)
print("\n--- OLS food model (no weather fixed effects) ---")
print(f"Intercept: {reg.intercept_:.6f}")
for name, coef in zip(features, reg.coef_):
    print(f"  {name}: {coef:.6f}")
print(f"R²: {reg.score(X, y):.4f}")

# Compute residuals
clean["food_pred_ols"] = reg.predict(X)
clean["food_resid_ols"] = clean["d_food"] - clean["food_pred_ols"]

# --- Step 2: Group residuals by (round, seed, step) ---
# If there's a weather component, it should be constant within each group
group_resid = clean.groupby(["round_id", "seed", "step"])["food_resid_ols"].agg(["mean", "std", "count"])
group_resid.columns = ["weather_mean", "weather_std", "n"]
group_resid = group_resid.reset_index()

print(f"\n--- Weather component (per group residual) ---")
print(f"Mean of group means: {group_resid['weather_mean'].mean():.6f}")
print(f"Std of group means: {group_resid['weather_mean'].std():.6f}")
print(f"Range: [{group_resid['weather_mean'].min():.4f}, {group_resid['weather_mean'].max():.4f}]")
print(f"Mean within-group std: {group_resid['weather_std'].mean():.6f}")
print(f"  (should be small if weather is truly constant within group)")

# --- Step 3: Now fit with fixed effects (demean by group) ---
clean = clean.merge(
    clean.groupby(["round_id", "seed", "step"])["d_food"].transform("mean").rename("group_mean_dfood"),
    left_index=True, right_index=True
)
# Actually let's do this properly
group_means = clean.groupby(["round_id", "seed", "step"])[features + ["d_food"]].transform("mean")
X_dm = clean[features].values - group_means[features].values
y_dm = clean["d_food"].values - group_means["d_food"].values

reg_fe = LinearRegression(fit_intercept=False).fit(X_dm, y_dm)
print("\n--- Fixed-effects food model (demeaned by round/seed/step) ---")
for name, coef in zip(features, reg_fe.coef_):
    print(f"  {name}: {coef:.6f}")
print(f"R² (within): {reg_fe.score(X_dm, y_dm):.4f}")

# --- Step 4: Get the weather component from fixed effects ---
# weather_{r,s,t} = mean(d_food - X*beta_fe) for each group
clean["food_pred_fe"] = clean[features].values @ reg_fe.coef_
clean["food_resid_fe"] = clean["d_food"] - clean["food_pred_fe"]
weather = clean.groupby(["round_id", "seed", "step"]).agg(
    weather=("food_resid_fe", "mean"),
    resid_std=("food_resid_fe", "std"),
    n=("food_resid_fe", "count"),
).reset_index()

print(f"\n--- Weather from FE model ---")
print(f"Mean: {weather['weather'].mean():.6f}")
print(f"Std: {weather['weather'].std():.6f}")
print(f"Range: [{weather['weather'].min():.4f}, {weather['weather'].max():.4f}]")
print(f"Mean within-group residual std: {weather['resid_std'].mean():.6f}")

# --- Step 5: Is weather actually "winter"? Does it depend on step? ---
weather_by_step = weather.groupby("step")["weather"].agg(["mean", "std"]).reset_index()

# --- PLOTS ---

# Plot 1: Histogram of weather component
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(weather["weather"], bins=60, edgecolor="black", alpha=0.7)
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_xlabel("Weather component (food delta residual)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of weather component\n(per round/seed/step)")

# Plot 2: Weather vs step
axes[1].scatter(weather["step"], weather["weather"], alpha=0.3, s=10)
axes[1].plot(weather_by_step["step"], weather_by_step["mean"], color="red", linewidth=2, label="Mean")
axes[1].fill_between(
    weather_by_step["step"],
    weather_by_step["mean"] - weather_by_step["std"],
    weather_by_step["mean"] + weather_by_step["std"],
    alpha=0.2, color="red"
)
axes[1].axhline(0, color="gray", linestyle="--")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Weather component")
axes[1].set_title("Weather component vs simulation step")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "winter_food_impact.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "winter_food_impact.png"), dpi=150)
print(f"\nSaved: {OUT}/winter_food_impact.pdf")

# Plot 3: Within-group consistency check
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(weather["n"], weather["resid_std"], alpha=0.3, s=10)
axes[0].set_xlabel("Group size (settlements per step)")
axes[0].set_ylabel("Within-group residual std")
axes[0].set_title("Residual consistency within (round, seed, step) groups")

# Plot 4: Weather across different round/seed combos
for (rid, seed), g in weather.groupby(["round_id", "seed"]):
    axes[1].plot(g["step"], g["weather"], alpha=0.3, linewidth=0.8)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Weather component")
axes[1].set_title("Weather traces per replay (all 45 replays)")
axes[1].axhline(0, color="red", linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "winter_food_detail.pdf"), dpi=150)
plt.savefig(os.path.join(OUT, "winter_food_detail.png"), dpi=150)
print(f"Saved: {OUT}/winter_food_detail.pdf")

# --- Summary statistics ---
print("\n" + "="*60)
print("CONCLUSION: Winter/Weather Impact on Food")
print("="*60)
print(f"The weather component has mean {weather['weather'].mean():.4f} and std {weather['weather'].std():.4f}")
print(f"This means on average, the combined effect (intercept + weather) is about {weather['weather'].mean():.4f}")
print(f"The weather varies from {weather['weather'].min():.4f} to {weather['weather'].max():.4f} per step")
print(f"Within each step, residual std is only {weather['resid_std'].mean():.4f}")
print(f"  => weather is nearly constant within each (round, seed, step)")
print(f"\nThe OLS intercept (base food production + mean winter) is: {reg.intercept_:.4f}")
print(f"Food production coefficients:")
for name, coef in zip(features, reg.coef_):
    print(f"  {name}: {coef:+.6f}")
