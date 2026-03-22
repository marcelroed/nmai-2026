"""Replay-based single-step evaluation: score simulator against known rollouts.

Usage:
    uv run src/astar_island/replay_eval.py [--infer] [--nsims N] [--rounds R1,R2,...] [--hidden]

Options:
    --infer         Run CMA-ES parameter inference per round before evaluation.
    --nsims N       Sims per step pair for evaluation (default 200).
    --rounds R,...  Comma-separated round IDs (default: all rounds with replays).
    --hidden        Optimize hidden state per step pair (slower but more accurate).
    --top N         Show top N worst step pairs per round (default 10).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

SIMULATOR_MANIFEST = Path(__file__).parent.parent.parent / "simulator" / "Cargo.toml"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROJECT_ROOT = Path(__file__).parent.parent.parent

CLASS_NAMES = ["Empty/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def _sim_cmd(*args: str) -> list[str]:
    """Build a cargo run command for the simulator."""
    return ["cargo", "run", "--release", "--manifest-path", str(SIMULATOR_MANIFEST), "--", *args]


def find_rounds_with_replays() -> list[str]:
    """Find all round IDs that have replay data."""
    rounds = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "analysis" / "replay_seed_index=0.json").exists():
            rounds.append(d.name)
    return rounds


def run_replay_infer(round_id: str, sims_per_pair: int = 30) -> str | None:
    """Run replay-infer and return the params JSON string."""
    result = subprocess.run(
        _sim_cmd("replay-infer", round_id, str(sims_per_pair)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"    inference failed: {result.stderr[:300]}")
        return None
    # stderr has progress, stdout has params JSON
    sys.stderr.write(result.stderr)
    return result.stdout.strip()


def run_replay_eval(
    round_id: str, n_sims: int, params_json: str | None = None
) -> tuple[list[dict], dict | None]:
    """Run replay-eval and return (pair_results, round_summary)."""
    cmd = _sim_cmd("replay-eval", round_id, str(n_sims))
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
        print(f"    eval failed: {result.stderr[:300]}")
        return [], None

    sys.stderr.write(result.stderr)

    pairs = []
    summary = None
    for line in result.stdout.strip().split("\n"):
        if line.startswith("PAIR "):
            pairs.append(json.loads(line[5:]))
        elif line.startswith("ROUND_SUMMARY "):
            summary = json.loads(line[14:])

    return pairs, summary


def print_round_report(round_id: str, pairs: list[dict], summary: dict | None, top_n: int):
    """Print a detailed report for one round."""
    short = round_id[:8]

    if not pairs:
        print(f"  [{short}] No results")
        return

    # Group by seed
    by_seed: dict[int, list[dict]] = {}
    for p in pairs:
        by_seed.setdefault(p["seed_index"], []).append(p)

    total_ll = sum(p["grid_ll"] for p in pairs)
    print(f"\n{'='*70}")
    print(f"Round {short}  |  total LL = {total_ll:.2f}  |  {len(pairs)} step pairs")
    print(f"{'='*70}")

    # Per-seed summary
    for seed_idx in sorted(by_seed.keys()):
        seed_pairs = by_seed[seed_idx]
        seed_ll = sum(p["grid_ll"] for p in seed_pairs)
        avg_modal = sum(p["modal_acc"] for p in seed_pairs) / len(seed_pairs)
        total_changed = sum(p["n_changed"] for p in seed_pairs)
        print(
            f"  seed {seed_idx}: LL={seed_ll:8.2f}  "
            f"avg modal acc={avg_modal:.1f}%  "
            f"total cells changed={total_changed}"
        )

    # Worst step pairs across all seeds
    worst = sorted(pairs, key=lambda p: p["grid_ll"])[:top_n]
    print(f"\n  Worst {top_n} step pairs:")
    for p in worst:
        n_ch = p["n_changed"]
        print(
            f"    seed {p['seed_index']} step {p['step']:2d}->{p['step']+1:2d}: "
            f"LL={p['grid_ll']:8.2f}  changed={n_ch:3d}  modal_acc={p['modal_acc']:.1f}%"
        )
        # Show worst cells for this pair
        for c in p["worst_cells"][:3]:
            from_name = CLASS_NAMES[c["from_class"]][:4]
            to_name = CLASS_NAMES[c["expected_class"]][:4]
            probs_str = " ".join(f"{CLASS_NAMES[i][:3]}={c['sim_probs'][i]:.2f}" for i in range(6) if c["sim_probs"][i] > 0.005)
            print(
                f"      ({c['y']:2d},{c['x']:2d}) {from_name}->{to_name}  "
                f"cell_ll={c['cell_ll']:.2f}  sim: [{probs_str}]"
            )

    # Aggregate transition analysis
    print(f"\n  Transition mismatches (aggregated across all step pairs):")
    trans_agg: dict[tuple[int, int], tuple[int, float]] = {}
    for p in pairs:
        for t in p["top_transitions"]:
            key = (t["from_class"], t["to_class"])
            real, sim = trans_agg.get(key, (0, 0.0))
            trans_agg[key] = (real + t["real_count"], sim + t["sim_avg"])

    trans_list = []
    for (fc, tc), (real, sim) in trans_agg.items():
        delta = real - sim
        if abs(delta) > 5:
            trans_list.append((fc, tc, real, sim, delta))
    trans_list.sort(key=lambda x: abs(x[4]), reverse=True)

    for fc, tc, real, sim, delta in trans_list[:15]:
        direction = "UNDER" if delta > 0 else "OVER"
        print(
            f"    {CLASS_NAMES[fc]:15s} -> {CLASS_NAMES[tc]:15s}: "
            f"real={real:5d}  sim={sim:7.1f}  delta={delta:+7.1f}  ({direction})"
        )

    # Settlement stat MAE summary
    mae_sums = [0.0, 0.0, 0.0, 0.0]
    n = 0
    for p in pairs:
        for i in range(4):
            mae_sums[i] += p["settlement_mae"][i]
        n += 1
    if n > 0:
        stat_names = ["population", "food", "wealth", "defense"]
        print(f"\n  Avg settlement stat MAE:")
        for i, name in enumerate(stat_names):
            print(f"    {name:12s}: {mae_sums[i] / n:.4f}")


def main():
    args = sys.argv[1:]

    use_infer = "--infer" in args
    use_hidden = "--hidden" in args
    n_sims = 200
    top_n = 10
    rounds = None

    if "--nsims" in args:
        idx = args.index("--nsims")
        n_sims = int(args[idx + 1])
    if "--top" in args:
        idx = args.index("--top")
        top_n = int(args[idx + 1])
    if "--rounds" in args:
        idx = args.index("--rounds")
        rounds = args[idx + 1].split(",")

    if rounds is None:
        rounds = find_rounds_with_replays()

    print(f"Replay evaluation: {n_sims} sims/pair, {len(rounds)} rounds, infer={use_infer}")
    print()

    all_pairs: dict[str, list[dict]] = {}
    all_summaries: dict[str, dict | None] = {}
    t0 = time.time()

    for round_id in rounds:
        short = round_id[:8]
        print(f"[{short}] Processing...")

        # Optionally infer params
        params_json = None
        if use_infer:
            print(f"  Inferring params...")
            params_json = run_replay_infer(round_id)
            if params_json:
                print(f"  Inferred params: {params_json[:60]}...")

        # Run evaluation
        print(f"  Evaluating ({n_sims} sims/pair)...")
        pairs, summary = run_replay_eval(round_id, n_sims, params_json)
        all_pairs[round_id] = pairs
        all_summaries[round_id] = summary

        print_round_report(round_id, pairs, summary, top_n)

    elapsed = time.time() - t0

    # Grand summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY")
    print(f"{'='*70}")
    print(f"| {'Round':8s} | {'Total LL':>10s} | {'Pairs':>5s} | {'Avg LL/pair':>11s} |")
    print(f"|{'─'*10}|{'─'*12}|{'─'*7}|{'─'*13}|")

    grand_ll = 0.0
    grand_pairs = 0
    for round_id in rounds:
        pairs = all_pairs.get(round_id, [])
        total_ll = sum(p["grid_ll"] for p in pairs)
        n_pairs = len(pairs)
        avg_ll = total_ll / n_pairs if n_pairs > 0 else 0
        grand_ll += total_ll
        grand_pairs += n_pairs
        print(f"| {round_id[:8]:8s} | {total_ll:10.2f} | {n_pairs:5d} | {avg_ll:11.4f} |")

    print(f"|{'─'*10}|{'─'*12}|{'─'*7}|{'─'*13}|")
    avg_ll = grand_ll / grand_pairs if grand_pairs > 0 else 0
    print(f"| {'TOTAL':8s} | {grand_ll:10.2f} | {grand_pairs:5d} | {avg_ll:11.4f} |")
    print(f"{'='*70}")
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
