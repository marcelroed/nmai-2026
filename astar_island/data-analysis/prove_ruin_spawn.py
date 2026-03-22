"""
Prove: newborn settlements on Ruin terrain ALWAYS have pop=0.400, def=0.150.

Reproduce:  python3 data-analysis/prove_ruin_spawn.py

Method:
  For every pair of consecutive frames in every replay, find settlements that
  are alive in frame N+1 at a position that had NO alive settlement in frame N.
  If the terrain at that position in frame N was Ruin (code 3), record (pop, def).
  Assert that the values are always exactly (0.4, 0.15).
"""

import json
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RUIN = 3


def load_replays():
    replays = []
    for round_dir in sorted(DATA_DIR.iterdir()):
        if not round_dir.is_dir():
            continue
        analysis_dir = round_dir / "analysis"
        if not analysis_dir.exists():
            continue
        for f in sorted(analysis_dir.glob("replay_seed_index=*.json")):
            with open(f) as fh:
                replays.append(json.load(fh))
    return replays


def main():
    replays = load_replays()
    print(f"Loaded {len(replays)} replays from {DATA_DIR}\n")

    per_round = {}  # round_id -> Counter of (pop, def)
    violations = []

    for replay in replays:
        rid = replay["round_id"]
        if rid not in per_round:
            per_round[rid] = Counter()
        frames = replay["frames"]

        for i in range(len(frames) - 1):
            fb, fa = frames[i], frames[i + 1]
            alive_before = {
                (s["x"], s["y"]) for s in fb["settlements"] if s["alive"]
            }

            for s in fa["settlements"]:
                if not s["alive"]:
                    continue
                pos = (s["x"], s["y"])
                if pos in alive_before:
                    continue
                terrain_before = fb["grid"][s["y"]][s["x"]]
                if terrain_before != RUIN:
                    continue

                pop = s["population"]
                defense = s["defense"]
                per_round[rid][(pop, defense)] += 1

                if pop != 0.4 or defense != 0.15:
                    violations.append(
                        {
                            "round": rid,
                            "seed": replay["seed_index"],
                            "step": fa["step"],
                            "pos": pos,
                            "pop": pop,
                            "def": defense,
                        }
                    )

    # ── Report ──────────────────────────────────────────────────────────────
    total = 0
    print(f"{'Round':>40s}  {'(0.4, 0.15)':>12s}  {'other':>6s}")
    print("-" * 65)
    for rid in sorted(per_round):
        c = per_round[rid]
        exact = c[(0.4, 0.15)]
        other = sum(v for k, v in c.items() if k != (0.4, 0.15))
        total += exact + other
        print(f"{rid}  {exact:>12d}  {other:>6d}")
    print("-" * 65)
    exact_total = sum(c[(0.4, 0.15)] for c in per_round.values())
    other_total = total - exact_total
    print(f"{'TOTAL':>40s}  {exact_total:>12d}  {other_total:>6d}")

    print(f"\nTotal ruin-born newborns: {total}")
    print(f"  Matching (0.4, 0.15): {exact_total} ({100*exact_total/total:.2f}%)")
    print(f"  Not matching:         {other_total}")

    if violations:
        print(f"\n*** VIOLATIONS ({len(violations)}) ***")
        for v in violations[:20]:
            print(f"  round={v['round'][:8]} seed={v['seed']} step={v['step']} "
                  f"pos={v['pos']} pop={v['pop']} def={v['def']}")
        ok = False
    else:
        print("\n==> CLAIM CONFIRMED: every ruin-born newborn has pop=0.400, def=0.150")
        ok = True

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
