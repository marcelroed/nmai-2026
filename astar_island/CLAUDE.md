# CLAUDE.md

## Project Structure

- `simulator/` — Rust simulator binary (`astar-simulator`). Build with `cargo build --release` from `astar_island/simulator/`.
- `src/astar_island/orchestrator.py` — Python end-to-end pipeline: fetch round data, run inference, predict, validate, submit.
- `src/astar_island/postprocess.py` — Post-processing of raw Monte Carlo predictions.
- `data/<round-id>/` — Cached round details, query results, analysis plots.
- `simulator_viz.html` — Visual documentation of all 7 simulation phases and 54 learnable parameters.

## Orchestrator CLI

```bash
uv run src/astar_island/orchestrator.py <round-id> <n-sims> [--infer] [--infer-budget N] [--validate]
```

- `<round-id>` — UUID of the competition round (omit to use the active round).
- `<n-sims>` — Number of Monte Carlo simulations per seed (default 2000).
- `--infer` — Run CMA-ES parameter inference on cached query observations before simulation.
- `--infer-budget N` — Multiply inference compute by N (scales population size and sims per candidate, default 1).
- `--validate` — Compare predictions against ground truth and save per-seed validation plots to `data/<round-id>/analysis/validation_seed_N.png`.

## Building the Simulator

```bash
cd simulator && cargo build --release
```

The orchestrator expects the binary at `simulator/target/release/astar-simulator`.

## Validation After Algorithm Changes

Every time you change the simulation algorithm, inference, or post-processing, you **must** validate on all 5 benchmark rounds:

```bash
echo N | uv run src/astar_island/orchestrator.py 36e581f1-73f8-453f-ab98-cbe3052b701b 50000 --infer --validate
echo N | uv run src/astar_island/orchestrator.py 3eb0c25d-28fa-48ca-b8e1-fc249e3918e9 50000 --infer --validate
echo N | uv run src/astar_island/orchestrator.py 324fde07-1670-4202-b199-7aa92ecb40ee 50000 --infer --validate
echo N | uv run src/astar_island/orchestrator.py fd3c92ff-3178-4dc9-8d9b-acf389b3982b 50000 --infer --validate
echo N | uv run src/astar_island/orchestrator.py f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb 50000 --infer --validate
```

After each run, copy the generated validation plots into:

```
track-best-algorithms/<algorithm-descriptor>/<round-id>/
```

Where `<algorithm-descriptor>` is a short kebab-case name for the change (e.g. `add-raid-distance-decay`, `tune-food-ocean`).

Then update `track-best-algorithms/RESULTS.md` — it contains a table tracking average scores for each (algorithm, round) pair. Mark the best score per round in **bold**.
