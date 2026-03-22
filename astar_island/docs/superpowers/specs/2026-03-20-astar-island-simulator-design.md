# Astar Island Simulator & Prediction Pipeline — Design Spec

## Problem Statement

Astar Island is a competition challenge where we observe a black-box Norse civilisation simulator through limited viewports and predict the final terrain probability distributions. Each round has hidden parameters controlling growth, conflict, and environmental dynamics. We have 50 viewport queries (15x15) on a 40x40 map across 5 seeds.

**Goal:** Build a faithful simulator, infer hidden parameters from queries, run Monte Carlo to produce maximum-score predictions.

## Prior Art & Data Analysis

We have 9 completed rounds with full ground truth (45 seed examples), including:
- Frame-by-frame replays (51 frames per seed, full 40x40 grid + all settlement stats)
- Ground truth probability distributions (40x40x6 tensors)
- 50 viewport query snapshots per round

### Confirmed Simulator Rules (constant across rounds)

| Rule | Evidence |
|------|----------|
| Ruins last exactly 1 step | 100% across all 45 replays |
| Ports require ocean adjacency | 100% of 1,060 observed port formations |
| Ports are permanent (never revert) | 0 reversions observed |
| Newborn on ruin: pop=0.400, def=0.150 | Exact across all rounds |
| Newborn on plains/forest: pop=0.500, def=0.200 | Exact across all rounds |
| 72% of new settlements at Chebyshev distance 1 | Consistent across rounds |
| Settlements spawn only on plains, ruin, or forest | 100% observed |
| Food clamped to [0, 1.0], defense to [0, 1.0] | Exact |
| The defense update in Step 2 is proportional to the population | See data-analysis/plot_defense_recovery.py |
| When a new settlement is started, the child starts with 20% of the parent's wealth | See prove_growth_wealth.png |
| Wealth stays constant by default | 
| When a ruin is reclaimed, the settlement starts with 20% of the parent's food | See prove_ruin_food.png |
| When a raid happens, the amount that is stolen transfers exactly from the victim to the attacker, no loss. | See defense_drop_isolated.png |


### Food Production Model (semi-constant)

```
food_delta = theta_base + theta_pop * pop + theta_feedback * food
             + theta_plains * n_adj_plains + theta_forest * n_adj_forest
             + theta_mountain * n_adj_mountain + theta_settlement * n_adj_settlements
food = clamp(food + food_delta, 0.0, 0.998)
```

Terrain coefficients (plains ~+0.011, forest ~+0.017) are relatively stable across rounds (CV < 0.26). The intercept (theta_base: 0.30-0.56) and some coefficients vary.

### Hidden Parameters (vary per round)

| Parameter | Range | CV |
|-----------|-------|-----|
| Food base intercept | 0.30–0.56 | 0.25 |
| Growth/expansion rate | 1.25–23.48 births/step | 0.78 |
| Death/conflict rate | 2.08–17.23 deaths/step | 0.65 |
| Owner change frequency | 11–363 total | 0.85 |
| Ruin->settlement reclaim prob | ~0.35–0.60 | ~0.30 |
| Ruin->forest prob | ~0.10–0.25 | ~0.30 |

### Key Dynamics

- **The Ruin Cycle**: Settlements die -> ruin (1 step) -> transition to settlement/plains/forest/port (probabilities are parameterized per round, cross-round averages: ~47%/35%/17%/1%)
- **Wealth depletion**: Wealth trends to zero across 50 steps (0.30 -> 0.01), partially offset by trade
- **Faction consolidation**: ~47 factions -> ~18 by step 50
- **Settlement growth**: ~47 initial -> ~184 by step 50 (4x, varies 0.1x to 9.6x by round)
- **Port acceleration**: Port formation rate increases over time
- **Forest only from ruin**: Forest appears exclusively on ruin tiles

### Important: Each Seed Has a Different Map

The 5 seeds within a round have **different terrain maps** (different ocean/mountain/forest/plains layouts and different numbers of initial settlements). Hidden parameters are shared across seeds, but dynamics play out differently on each map. Viewport observations from different seeds cannot be pooled as if from the same map — the CMA-ES scoring function must simulate each seed on its own map and compare only against viewports for that specific seed.

## Architecture

### System Components

```
simulator/              # New Rust crate (all compute-heavy work)
  src/
    lib.rs              # Public API + PyO3 bindings
    world.rs            # Grid + settlement state
    step.rs             # Per-step simulation (6 phases)
    params.rs           # Hidden parameter struct + bounds
    food.rs             # Food production model
    growth.rs           # Settlement expansion logic
    conflict.rs         # Raiding + takeover
    trade.rs            # Port-to-port trade
    winter.rs           # Seasonal food loss + collapse
    environment.rs      # Ruin reclamation, forest growth
    calibrate.rs        # Fit structural constants from replays
    inference.rs        # CMA-ES parameter search
    montecarlo.rs       # Run N sims -> probability distribution
    io.rs               # JSON serialization for data exchange
    main.rs             # CLI entry point

src/astar_island/       # Python orchestrator
  data/                 # Existing API client (keep)
  orchestrator.py       # End-to-end pipeline
  query_planner.py      # Strategic viewport placement
  submit/               # Submission (enhance with floor + renormalize)
```

### Why Rust + Python

- **Rust**: A single 50-step simulation on 40x40 grid with ~200 settlements should take ~1-2ms. This enables 500K-1M simulations per minute on 64 cores. Uses Rayon for thread-level parallelism. Spatial indexing (grid bucketing) keeps conflict phase O(n*k) instead of O(n^2).
- **Python**: Orchestration, API calls, submission. Reuses existing auth infrastructure.
- **Integration**: PyO3 for the CMA-ES inner loop (avoids serialization overhead). CLI for debugging and one-off runs.

## Simulator Design

### Terrain Types

```rust
enum TerrainType {
    Empty      = 0,   // Class 0 — generic empty cell
    Settlement = 1,   // Class 1
    Port       = 2,   // Class 2
    Ruin       = 3,   // Class 3
    Forest     = 4,   // Class 4
    Mountain   = 5,   // Class 5
    Ocean      = 10,  // Class 0 — impassable water, borders map
    Plains     = 11,  // Class 0 — flat land, buildable
}
```

For predictions, Ocean/Plains/Empty all map to class 0. The internal representation must distinguish them because:
- Food model: `n_adj_plains` counts only Plains (11), not Ocean (10) or Empty (0)
- Ocean is impassable and determines port eligibility
- Plains is the primary buildable terrain

### Intentional Simplifications

- **tech_level** is mentioned in the official docs as a settlement property but is never exposed in the API or replay data. We omit it entirely. Trade's "technology diffusion" effect is modeled implicitly through the food/wealth bonuses.
- **Longship ownership** is also not exposed. We assume all ports have longship capability (ports extend raid range via `longship_range_bonus` in the conflict phase).

### State

```rust
struct World {
    width: usize,
    height: usize,
    grid: Vec<Vec<TerrainType>>,       // 40x40
    settlements: Vec<Settlement>,
    // Spatial index: grid-cell -> settlement indices for O(1) neighbor lookups
    spatial_index: Vec<Vec<Vec<usize>>>,
    rng: StdRng,                        // seeded for reproducibility
}

struct Settlement {
    x: u16,
    y: u16,
    population: f32,
    food: f32,
    wealth: f32,
    defense: f32,
    has_port: bool,                     // ports implicitly have longship capability (extends raid range)
    alive: bool,
    owner_id: u16,
}

struct Params {
    // Food production
    food_base: f32,              // 0.30 - 0.56
    food_pop_coeff: f32,         // ~-0.11
    food_feedback: f32,          // ~-0.45
    food_plains: f32,            // ~+0.011
    food_forest: f32,            // ~+0.017
    food_mountain: f32,          // ~-0.003
    food_settlement: f32,        // ~-0.007

    // Population dynamics
    pop_growth_rate: f32,        // per-step growth (food-dependent)
    pop_max: f32,                // soft cap

    // Defense dynamics
    defense_recovery_rate: f32,  // per-step recovery toward baseline

    // Growth (settlement expansion)
    growth_prob: f32,            // base per-settlement per-step spawn probability
    spawn_distance_max: f32,     // ~2-5

    // Conflict
    raid_prob: f32,              // probability of raid per pair
    raid_range: f32,             // base range
    longship_range_bonus: f32,   // extra range for ports with longships
    aggression: f32,             // food-desperation multiplier
    takeover_threshold: f32,     // defense below which takeover happens
    raid_damage: f32,            // defense/pop damage per raid

    // Trade
    trade_range: f32,            // max port-to-port trade distance
    trade_food_bonus: f32,       // food generated per trade pair
    trade_wealth_bonus: f32,     // wealth generated per trade pair

    // Winter
    winter_severity: f32,        // food loss per step
    collapse_threshold: f32,     // food below which settlement dies

    // Environment
    ruin_to_settlement: f32,     // reclaim probability (0.35-0.60)
    ruin_to_forest: f32,         // forest growth probability (0.10-0.25)
    // ruin_to_plains = 1 - ruin_to_settlement - ruin_to_forest - ruin_to_port
    reclaim_range: f32,          // max distance for reclaim

    // Initial settlement stats (unknown from API, inferred)
    init_pop_mean: f32,          // mean initial population
    init_food_mean: f32,         // mean initial food
    init_wealth_mean: f32,       // mean initial wealth
    init_defense_mean: f32,      // mean initial defense
}
```

### Initial Settlement Stats

The API only provides `{x, y, has_port, alive}` for initial settlements — population, food, wealth, defense are hidden. Strategy for live rounds:

1. **Learn empirical distributions** from 45 replays (step 0 stats): pop ~1.0, food ~0.55, wealth ~0.30, defense ~0.40.
2. **Parameterize with 4 mean values** (init_pop_mean, init_food_mean, etc.) in Params struct. All settlements in a round share these means with small random noise.
3. During calibration, validate that the exact initial values matter less than the hidden dynamics parameters (hypothesis: the system is attracted to its steady state quickly, so initial conditions wash out).

### Settlement Processing Order

Within each phase, settlements are processed in a random shuffled order (re-shuffled each step). This will be validated during calibration by comparing random-order vs fixed-order variants against replay data.

### Faction-Based Interaction Rules

- Same-faction settlements do NOT raid each other
- Same-faction settlements CAN trade (if both are ports)
- Reclaimed ruins inherit the owner_id of the nearest allied settlement
- Population dispersal on death goes to nearby same-faction settlements

### Phase Ordering Note

The official docs describe 5 phases: "growth, conflict, trade, harsh winters, and environmental change." Our decomposition splits growth into sub-phases (food production, pop/defense dynamics, then expansion). This decomposition is a hypothesis — during calibration (implementation step 2), we must test this ordering against alternatives (e.g., food computed within the growth pass) and pick whichever produces better frame-by-frame replay fidelity.

### Per-Step Logic (50 steps)

**Phase 1: Food Production** (deterministic per settlement)
- For each alive settlement, compute food_delta from linear model
- Clamp food to [0, 0.998]

**Phase 2: Population & Defense Dynamics**
- Population: `pop += pop_growth_rate * food * (1 - pop/pop_max)` (logistic growth, food-dependent)
- Defense: `defense += defense_recovery_rate * (baseline - defense)` (recovery toward baseline)
- Clamp both to valid ranges

**Phase 3: Growth** (stochastic, settlement expansion)
- For each alive settlement, compute spawn probability: `p = growth_prob * pop * food`
- If spawning: pick random valid target within Chebyshev distance (weighted by empirical distance distribution: 72% dist 1, 21% dist 2, 7% dist 3+)
- Target terrain must be plains, ruin, or forest
- Create child settlement with terrain-dependent stats (ruin: pop=0.4/def=0.15, plains/forest: pop=0.5/def=0.2)
- Parent loses population equal to child's; if parent pop <= 0, parent dies -> ruin
- Coastal ruin targets (adjacent to ocean) become ports directly
- Child inherits parent's owner_id
- Only spawn if parent has enough population (pop > child_pop)

**Phase 4: Conflict** (stochastic, raiding + takeover)
- Use spatial index for efficient neighbor lookup
- For each pair of settlements with different owner_id within raid_range (+ longship_range_bonus if attacker has_longship):
  - Compute raid probability: `p = raid_prob * (1 + aggression * (1 - attacker.food))`
  - If raiding: defender.defense -= raid_damage, defender.pop *= 0.85, attacker.wealth += stolen
  - Consecutive raids stack damage within the same step
  - If defender.defense < takeover_threshold: either takeover (owner changes, 60%) or death -> ruin (40%)

**Phase 5: Trade** (between ports)
- For each pair of allied port settlements within trade_range:
  - Both gain food: `food += trade_food_bonus`
  - Both gain wealth: `wealth += trade_wealth_bonus`
  - Tech diffuses (modeled implicitly through stat boosts)
- Non-allied ports do not trade

**Phase 6: Winter** (all settlements)
- All settlements: `food -= winter_severity`
- If food < collapse_threshold: settlement dies -> becomes ruin
  - Grid cell set to Ruin (terrain code 3)
  - Population disperses to nearby same-faction settlements (within distance 3)
  - Settlement marked as dead

**Phase 7: Environment** (ruin cells)
- For each ruin cell, sample transition:
  - If nearby thriving settlement (within reclaim_range): `P(settlement) = ruin_to_settlement`
  - If adjacent to ocean AND nearby settlement: `P(port) = ruin_to_settlement * 0.1` (small fraction of reclaims become ports)
  - `P(forest) = ruin_to_forest`
  - `P(plains) = 1 - P(settlement) - P(port) - P(forest)`
  - New settlement inherits owner from nearest same-faction settlement

## Parameter Inference

### Calibration (offline, one-time)

Using 45 replays with full observability:

1. **Fit structural constants**: Linear regression for food model, empirical distributions for spawn distances, exact values for newborn stats.
2. **Fit population/defense dynamics**: Regression on frame-to-frame changes for pop and defense, conditioned on food/combat state.
3. **Fit trade effects**: Identify food/wealth gains in port pairs that are allied and within range.
4. **Per-round parameter fitting**: For each historical round, fit hidden params to minimize replay prediction error. This gives us 9 data points of (true_params, observed_dynamics).
5. **Prior construction**: Build prior distributions over hidden params from the 9 fitted rounds.

### Live Inference (per new round)

**Input**: Initial states (5 seeds, each a different map), 50 viewport query results.

**Method: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**

1. Initialize with prior mean/covariance from historical rounds
2. For each candidate theta:
   a. For each seed: run simulator K=20 times from that seed's initial state
   b. Extract simulated terrain in viewport regions corresponding to that seed's queries
   c. Compare simulated terrain distributions to observed query terrain distributions
   d. Score = negative sum of per-cell log-likelihood across all viewports (aligns with KL-based scoring metric)
3. CMA-ES iterates ~100-500 generations with population size ~50-200
4. Total simulations: ~500 generations * 100 candidates * 20 runs * 5 seeds = 50M sims
5. On 64 cores at ~750K sims/min = ~67 minutes. On 256 cores = ~17 minutes.

**Comparison metric per viewport**: For each cell in the 15x15 viewport, compute log-likelihood of observed terrain type under the simulated empirical distribution (from K runs). Sum across all cells and all viewports. This directly aligns with the KL divergence scoring metric, rather than chi-squared which is unreliable for rare terrain classes.

### Monte Carlo Prediction

Once hidden params are inferred:

1. Run simulator N=2000 times per seed with the best-fit params
2. For each cell, count terrain type frequencies across N runs
3. Normalize to probabilities
4. Apply floor of 0.01 per class, renormalize to sum to 1.0
5. Output: 40x40x6 tensor per seed

## Query Strategy

50 queries across 5 seeds on 40x40 map:

### Allocation

- **Coverage pass (9 queries, ~2 per seed for 2-3 seeds)**: Tile the map with non-overlapping 15x15 viewports. 3x3 grid covers 45x45, so 9 viewports cover most of the 40x40 map. Allocate to 2 seeds for broader coverage.
- **Repeated observations (41 queries)**: Same viewport positions, different sim seeds. Critical for estimating stochastic variance. Focus on 2-3 seeds with most queries for statistical power.
- **Seed focus**: Allocate ~15-20 queries to 2 primary seeds, ~5-8 to 2 secondary seeds, ~2-4 to 1 seed. More repeats on fewer seeds gives better parameter signal.

### Viewport Placement

Prioritize regions with settlements (dynamic cells). The initial state tells us where settlements are. Place viewports to maximize coverage of settlement-rich areas.

## Validation Protocol

### Oracle Test (per historical round)

Fit params from full replay -> run 1000 sims -> compare to ground truth distribution.
**Target**: Mean entropy-weighted KL divergence < 0.05.

### End-to-End Test (per historical round)

Use only 50 viewport queries -> infer params -> run Monte Carlo -> compare to ground truth.
**Target**: Match or beat historical submitted scores.

### Ablation Studies

- Monte Carlo sample count: 100, 500, 1000, 2000, 5000
- CMA-ES iterations: 50, 100, 200, 500
- Query allocation strategies
- Number of hidden parameters (start simple, add complexity)

## Parallelism

- **Rayon** for intra-machine thread parallelism (all cores)
- **PyO3** for Rust-Python integration in the CMA-ES loop (no JSON serialization overhead)
- **CLI batch mode** for debugging and multi-machine distribution
- **Target throughput**: 500K-1M simulations per minute per machine (64 cores)

## Fallback Strategy

If the full simulator + CMA-ES pipeline does not produce results in time for a round:

1. **Empirical fallback**: Use terrain frequencies from viewport observations directly. For cells outside viewports, use the initial state terrain with uniform noise for dynamic cells.
2. **Prior-only fallback**: Use the mean of historical hidden parameters (no inference), run Monte Carlo, submit. Even with wrong parameters, this should beat uniform.
3. **Uniform baseline**: Submit uniform 1/6 predictions with terrain constraints (ocean=1.0 empty, mountain=1.0 mountain). This is the absolute floor — always submit something.

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Simulator not faithful enough | Validate frame-by-frame on 9 historical rounds. Iterate rules. |
| Parameter inference doesn't converge | Start with fewer params (5-7), add complexity. Use replay-derived priors. |
| 50 queries insufficient for inference | Optimize query placement on historical data. Leverage initial state heavily. |
| Compute budget exceeded | Profile early. Use spatial indexing. Budget 2x estimated time. |
| Probability floor destroys signal | Use 0.01 floor (per docs). Test sensitivity. |
| Initial settlement stats unknown | Parameterize with 4 mean values. Validate sensitivity. |
| Settlement processing order matters | Test random vs fixed order during calibration. |

## Implementation Order

1. Rust simulator core (world, step, params) — validate against replay frames
2. Calibration module — fit structural constants from replays
3. Monte Carlo runner — generate probability distributions
4. Oracle validation — compare to ground truth on historical rounds
5. Parameter inference (CMA-ES) — infer hidden params from viewport queries
6. Python orchestrator — API integration, query planning, submission
7. End-to-end validation — full pipeline on historical rounds
8. Query strategy optimization — tune viewport placement
