#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench_suite as bs


def row_key(row: dict) -> tuple[int, int, int, int]:
    return (
        int(row["forest_height"]),
        int(row["rounds"]),
        int(row["batch_size"]),
        int(row["seed"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark comparison")
    parser.add_argument("--base", required=True, help="Path to baseline module")
    parser.add_argument("--candidate", required=True, help="Path to candidate module")
    parser.add_argument("--cases", default="10,16,256")
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--db", default="experiments/bench_results.sqlite")
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    cases = bs.expand_cases(args.cases, args.seeds)
    base_rows = bs.run_jobs(args.base, cases, args.workers)
    cand_rows = bs.run_jobs(args.candidate, cases, args.workers)

    base_map = {row_key(r): r for r in base_rows}
    cand_map = {row_key(r): r for r in cand_rows}

    deltas: list[int] = []
    wins = 0
    ties = 0
    losses = 0

    keys = sorted(base_map.keys())
    print("Case-by-case deltas (candidate - base):")
    for key in keys:
        b = base_map[key]
        c = cand_map[key]
        if not b["correct"] or not c["correct"]:
            print(f"FAIL key={key} base_ok={b['correct']} cand_ok={c['correct']}")
            continue
        delta = int(c["cycles"]) - int(b["cycles"])
        deltas.append(delta)
        if delta < 0:
            wins += 1
        elif delta > 0:
            losses += 1
        else:
            ties += 1
        print(f"H={key[0]} R={key[1]} B={key[2]} seed={key[3]} delta={delta}")

    if deltas:
        print(
            "Summary: "
            f"mean_delta={statistics.mean(deltas):.3f} "
            f"best={min(deltas)} worst={max(deltas)} "
            f"wins={wins} ties={ties} losses={losses}"
        )
    else:
        print("No comparable successful cases.")

    if not args.no_db:
        bs.write_sqlite(args.db, "ab_base", args.base, base_rows)
        bs.write_sqlite(args.db, "ab_candidate", args.candidate, cand_rows)

    any_fail = any((not r["correct"]) or int(r["cycles"]) < 0 for r in base_rows + cand_rows)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
