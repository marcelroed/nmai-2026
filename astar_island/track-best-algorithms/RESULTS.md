# Algorithm Benchmark Results

Scores are the average post-processed competition score (0–100) across all validation seeds for each round.
**Bold** = best score for that round across all algorithms tested.

| Algorithm |  36e581f1 | 3eb0c25d  | 324fde07  | fd3c92ff  | f1dac9a9  |    Avg    |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| base      | **76.88** | **88.06** | **84.31** | **83.42** | **90.49** | **84.60** |

## Round Reference

| Short ID | Full Round ID |
|----------|---------------|
| 36e581f1 | `36e581f1-73f8-453f-ab98-cbe3052b701b` |
| 3eb0c25d | `3eb0c25d-28fa-48ca-b8e1-fc249e3918e9` |
| 75e625c3 | `75e625c3-60cb-4392-af3e-c86a98bde8c2` |
| fd3c92ff | `fd3c92ff-3178-4dc9-8d9b-acf389b3982b` |
| f1dac9a9 | `f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb` |

## How to Add a Row

1. Make your algorithm change and rebuild the simulator if needed (`cd simulator && cargo build --release`).
2. Run all 5 benchmark rounds (50k sims each):
   ```bash
   uv run src/astar_island/orchestrator.py 36e581f1-73f8-453f-ab98-cbe3052b701b 50000 --infer --validate
   uv run src/astar_island/orchestrator.py 3eb0c25d-28fa-48ca-b8e1-fc249e3918e9 50000 --infer --validate
   uv run src/astar_island/orchestrator.py 75e625c3-60cb-4392-af3e-c86a98bde8c2 50000 --infer --validate
   uv run src/astar_island/orchestrator.py fd3c92ff-3178-4dc9-8d9b-acf389b3982b 50000 --infer --validate
   uv run src/astar_island/orchestrator.py f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb 50000 --infer --validate
   ```
3. Copy validation plots into `track-best-algorithms/<algorithm-descriptor>/<round-id>/`.
4. Add a row to the table above with the average scores from each run.
5. **Bold** the best score in each column.
