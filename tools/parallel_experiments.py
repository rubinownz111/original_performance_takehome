#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench_suite as bs


def _run_experiment(job: dict[str, Any]) -> dict[str, Any]:
    label = str(job["label"])
    module = str(job["module"])
    cases_spec = str(job["cases"])
    seeds_spec = str(job["seeds"])
    workers = int(job["workers"])
    db_path = str(job["db"])
    write_db = bool(job["write_db"])

    cases = bs.expand_cases(cases_spec, seeds_spec)
    rows = bs.run_jobs(module, cases, workers)
    summary = bs.summarize(rows)
    if write_db:
        bs.write_sqlite(db_path, label, module, rows)

    return {
        "label": label,
        "module": module,
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run many experiment modules in parallel")
    parser.add_argument(
        "--experiments-file",
        required=True,
        help="JSON array: [{\"label\":...,\"module\":...}, ...]",
    )
    parser.add_argument("--cases", default="10,16,256")
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--workers-per-exp", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--db", default="experiments/bench_results.sqlite")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    exp_path = Path(args.experiments_file)
    experiments = json.loads(exp_path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for item in experiments:
        jobs.append(
            {
                "label": item["label"],
                "module": str(Path(item["module"]).resolve()),
                "cases": args.cases,
                "seeds": args.seeds,
                "workers": args.workers_per_exp,
                "db": args.db,
                "write_db": not args.no_db,
            }
        )

    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
        for result in pool.map(_run_experiment, jobs):
            results.append(result)

    ranked = sorted(
        results,
        key=lambda r: (r["summary"]["mean_cycles"] if r["summary"]["mean_cycles"] is not None else 10**9),
    )

    print("Leaderboard (lower mean cycles is better):")
    for r in ranked:
        s = r["summary"]
        print(
            f"{r['label']}: mean={s['mean_cycles']} best={s['best_cycles']} "
            f"worst={s['worst_cycles']} good={s['good']}/{s['total']}"
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ranked": ranked}, indent=2, sort_keys=True), encoding="utf-8")

    any_fail = any(r["summary"]["bad"] > 0 for r in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
