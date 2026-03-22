"""Benchmark iteration tool: run all 5 rounds, score, diagnose, plot.

Usage:
    uv run src/astar_island/bench.py [--name <algorithm-name>] [--nsims N] [--no-infer] [--quick]

Options:
    --name <name>   Algorithm descriptor (e.g. "tune-food-ocean"). If given, copies
                    plots to track-best-algorithms/<name>/ and updates RESULTS.md.
    --nsims N       Monte Carlo sims per seed (default 50000, or 5000 with --quick).
    --no-infer      Skip CMA-ES parameter inference (inference runs by default).
    --quick         Use 5000 sims for fast iteration (overridden by explicit --nsims).
"""

import contextlib
import io
import json
import math
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from astar_island.orchestrator import (
    CLASS_NAMES,
    DATA_DIR,
    competition_score,
    per_class_weighted_kl,
    plot_validation,
    score_prediction,
)
from astar_island.postprocess import postprocess_predictions

warnings.filterwarnings("ignore", category=RuntimeWarning)

SIMULATOR_MANIFEST = Path(__file__).parent.parent.parent / "simulator" / "Cargo.toml"


def _sim_cmd(*args: str) -> list[str]:
    """Build a cargo run command for the simulator."""
    return ["cargo", "run", "--release", "--manifest-path", str(SIMULATOR_MANIFEST), "--", *args]

BENCHMARK_ROUNDS = [
    "36e581f1-73f8-453f-ab98-cbe3052b701b",
    "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9",
    "324fde07-1670-4202-b199-7aa92ecb40ee",
    "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
]

SHORT_IDS = {r: r[:8] for r in BENCHMARK_ROUNDS}

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_round_details(round_id: str) -> dict:
    detail_path = DATA_DIR / round_id / "details.json"
    if not detail_path.exists():
        raise FileNotFoundError(f"No cached details for {round_id}. Run the orchestrator first.")
    return json.loads(detail_path.read_text())


def run_inference(round_id: str) -> str | None:
    result = subprocess.run(
        _sim_cmd("infer", round_id),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"    inference failed: {result.stderr[:200]}")
        return None
    return result.stdout.strip()


def run_montecarlo(round_id: str, n_sims: int, params_json: str | None = None) -> dict[int, list]:
    cmd = _sim_cmd("montecarlo", round_id, str(n_sims))
    if params_json:
        cmd.append(params_json)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"    simulator failed: {result.stderr[:300]}")
        return {}
    predictions = {}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("SEED_"):
            parts = line.split(":", 1)
            seed_idx = int(parts[0].split("_")[1])
            predictions[seed_idx] = json.loads(parts[1])
    return predictions


def load_ground_truth(round_id: str) -> dict[int, list]:
    gt = {}
    analysis_dir = DATA_DIR / round_id / "analysis"
    for gt_path in sorted(analysis_dir.glob("ground_truth_seed_index=*.json")):
        seed_idx = int(gt_path.stem.split("=")[1])
        data = json.loads(gt_path.read_text())
        gt[seed_idx] = data["ground_truth"]
    return gt


def diagnose_round(
    round_id: str,
    raw_predictions: dict[int, list],
    predictions: dict[int, list],
    ground_truths: dict[int, list],
    initial_states: list[dict],
    generate_plots: bool = True,
) -> dict:
    """Score and diagnose one round. Returns per-seed results."""
    results = {}
    for seed_idx in sorted(predictions.keys()):
        if seed_idx not in ground_truths:
            continue
        gt = ground_truths[seed_idx]
        raw_kl = score_prediction(raw_predictions[seed_idx], gt)
        raw_cs = competition_score(raw_kl)
        pp_kl = score_prediction(predictions[seed_idx], gt)
        pp_cs = competition_score(pp_kl)

        pp_per_class = per_class_weighted_kl(
            np.array(predictions[seed_idx]), np.array(gt)
        )

        results[seed_idx] = {
            "raw_kl": raw_kl,
            "raw_score": raw_cs,
            "pp_kl": pp_kl,
            "pp_score": pp_cs,
            "per_class_kl": pp_per_class,
        }
    return results


def print_round_diagnostics(round_id: str, results: dict):
    short = SHORT_IDS.get(round_id, round_id[:8])
    scores = []
    total_per_class = np.zeros(6)
    n = 0

    for seed_idx in sorted(results.keys()):
        r = results[seed_idx]
        scores.append(r["pp_score"])
        total_per_class += r["per_class_kl"]
        n += 1

        # One-line per seed
        class_str = "  ".join(
            f"{CLASS_NAMES[c][:4]}={r['per_class_kl'][c]:.4f}" for c in range(6)
        )
        print(
            f"    seed {seed_idx}: raw={r['raw_score']:5.1f}  pp={r['pp_score']:5.1f}  | {class_str}"
        )

    if n > 0:
        avg_score = sum(scores) / n
        avg_per_class = total_per_class / n
        print(f"    --- {short} avg score: {avg_score:.2f}/100 ---")

        # Highlight worst class
        worst_class = int(np.argmax(avg_per_class))
        print(
            f"    worst class: {CLASS_NAMES[worst_class]} "
            f"(avg KL contribution = {avg_per_class[worst_class]:.4f})"
        )
        return avg_score
    return 0.0


def print_summary_table(round_scores: dict[str, float]):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = "| Round    | Score  |"
    sep = "|----------|--------|"
    print(header)
    print(sep)
    for round_id in BENCHMARK_ROUNDS:
        short = SHORT_IDS.get(round_id, round_id[:8])
        score = round_scores.get(round_id, 0.0)
        print(f"| {short} | {score:6.2f} |")
    avg = sum(round_scores.values()) / len(round_scores) if round_scores else 0.0
    print(sep)
    print(f"| {'AVG':8s} | {avg:6.2f} |")
    print("=" * 70)
    return avg


def print_class_breakdown(all_per_class: dict[str, np.ndarray]):
    """Print per-class KL breakdown averaged across all rounds."""
    print("\nPer-class avg KL contribution (across all rounds):")
    combined = np.zeros(6)
    n = 0
    for per_class in all_per_class.values():
        combined += per_class
        n += 1
    if n > 0:
        combined /= n
    ranked = sorted(range(6), key=lambda c: combined[c], reverse=True)
    for c in ranked:
        bar = "#" * int(combined[c] * 200)
        print(f"  {CLASS_NAMES[c]:15s}  {combined[c]:.4f}  {bar}")


def save_to_tracking(name: str, round_scores: dict[str, float]):
    """Copy plots and update RESULTS.md."""
    tracking_dir = PROJECT_ROOT / "track-best-algorithms"
    tracking_dir.mkdir(exist_ok=True)

    for round_id in BENCHMARK_ROUNDS:
        src_dir = DATA_DIR / round_id / "analysis"
        dst_dir = tracking_dir / name / round_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        for png in src_dir.glob("validation_seed_*.png"):
            shutil.copy2(png, dst_dir / png.name)

    # Update RESULTS.md
    results_path = tracking_dir / "RESULTS.md"
    if results_path.exists():
        content = results_path.read_text()
    else:
        content = ""

    # Build the new row
    scores_str = " | ".join(
        f"{round_scores.get(r, 0.0):9.2f}" for r in BENCHMARK_ROUNDS
    )
    avg = sum(round_scores.values()) / len(round_scores) if round_scores else 0.0
    new_row = f"| {name:9s} | {scores_str} | {avg:9.2f} |"

    # Check if this algorithm already has a row
    lines = content.split("\n")
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"| {name}") or line.startswith(f"| {name} "):
            lines[i] = new_row
            updated = True
            break

    if not updated:
        # Find the last table row (before empty line or ## section)
        insert_idx = None
        for i, line in enumerate(lines):
            if line.startswith("|") and not line.startswith("|-") and not line.startswith("| Algorithm") and not line.startswith("| Round") and not line.startswith("| Short"):
                insert_idx = i + 1
        if insert_idx is not None:
            lines.insert(insert_idx, new_row)
        else:
            lines.append(new_row)

    results_path.write_text("\n".join(lines))
    print(f"\nResults saved to {tracking_dir / name}/")
    print(f"RESULTS.md updated.")


def main():
    args = sys.argv[1:]

    name = None
    n_sims = None
    use_infer = "--no-infer" not in args
    quick = "--quick" in args

    if "--name" in args:
        idx = args.index("--name")
        name = args[idx + 1]
    if "--nsims" in args:
        idx = args.index("--nsims")
        n_sims = int(args[idx + 1])

    if n_sims is None:
        n_sims = 5000 if quick else 50000

    print(f"Benchmark: {n_sims} sims/seed, infer={use_infer}")
    print(f"Rounds: {len(BENCHMARK_ROUNDS)}")
    if name:
        print(f"Algorithm: {name}")
    print()

    round_scores: dict[str, float] = {}
    all_per_class: dict[str, np.ndarray] = {}
    t0 = time.time()

    for round_id in BENCHMARK_ROUNDS:
        short = SHORT_IDS[round_id]
        print(f"[{short}] Running...")

        # Load details and ground truth
        try:
            details = load_round_details(round_id)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue
        ground_truths = load_ground_truth(round_id)
        if not ground_truths:
            print(f"    SKIP: no ground truth")
            continue

        # Optionally infer
        params_json = None
        if use_infer:
            print(f"    inferring params...")
            params_json = run_inference(round_id)
            if params_json:
                print(f"    params: {params_json[:80]}...")

        # Run MC
        print(f"    running {n_sims} sims...")
        raw_predictions = run_montecarlo(round_id, n_sims, params_json)
        if not raw_predictions:
            print(f"    SKIP: simulator produced no output")
            continue

        # Postprocess (suppress internal prints)
        with contextlib.redirect_stdout(io.StringIO()):
            predictions = postprocess_predictions(
                raw_predictions, details["initial_states"], round_id
            )

        # Diagnose
        results = diagnose_round(
            round_id, raw_predictions, predictions, ground_truths,
            details["initial_states"], generate_plots=True,
        )

        avg_score = print_round_diagnostics(round_id, results)
        round_scores[round_id] = avg_score

        # Aggregate per-class KL
        per_class_sum = np.zeros(6)
        n = 0
        for r in results.values():
            per_class_sum += r["per_class_kl"]
            n += 1
        if n > 0:
            all_per_class[round_id] = per_class_sum / n

        print()

    elapsed = time.time() - t0

    # Summary
    avg = print_summary_table(round_scores)
    print_class_breakdown(all_per_class)
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save tracking
    if name:
        save_to_tracking(name, round_scores)

    return avg


if __name__ == "__main__":
    main()
