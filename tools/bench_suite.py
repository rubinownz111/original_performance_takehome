#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import csv
import dataclasses
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problem import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2


@dataclasses.dataclass(frozen=True)
class BenchCase:
    forest_height: int
    rounds: int
    batch_size: int
    seed: int


def parse_triplet(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid case '{text}', expected H,R,B")
    return int(parts[0]), int(parts[1]), int(parts[2])


def parse_cases(spec: str) -> list[tuple[int, int, int]]:
    items = [s.strip() for s in spec.split(";") if s.strip()]
    if not items:
        raise ValueError("At least one case is required")
    return [parse_triplet(item) for item in items]


def parse_seeds(spec: str) -> list[int]:
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if not items:
        raise ValueError("At least one seed is required")
    return [int(s) for s in items]


def expand_cases(case_spec: str, seed_spec: str) -> list[BenchCase]:
    triples = parse_cases(case_spec)
    seeds = parse_seeds(seed_spec)
    out: list[BenchCase] = []
    for h, r, b in triples:
        for seed in seeds:
            out.append(BenchCase(h, r, b, seed))
    return out


def _load_module(module_path: str):
    path = Path(module_path).resolve()
    name = f"bench_mod_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_case(job: dict[str, Any]) -> dict[str, Any]:
    case = BenchCase(**job["case"])
    module_path = job["module_path"]
    started = time.perf_counter()
    try:
        mod = _load_module(module_path)

        random.seed(case.seed)
        forest = Tree.generate(case.forest_height)
        inp = Input.generate(forest, case.batch_size, case.rounds)
        mem = build_mem_image(forest, inp)

        kb = mod.KernelBuilder()
        with contextlib.redirect_stdout(io.StringIO()):
            kb.build_kernel(
                forest.height,
                len(forest.values),
                len(inp.indices),
                case.rounds,
            )

        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        for ref_mem in reference_kernel2(mem):
            pass
        inp_values_p = ref_mem[6]
        correct = (
            machine.mem[inp_values_p:inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p:inp_values_p + len(inp.values)]
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "forest_height": case.forest_height,
            "rounds": case.rounds,
            "batch_size": case.batch_size,
            "seed": case.seed,
            "cycles": int(machine.cycle),
            "correct": bool(correct),
            "duration_ms": elapsed_ms,
            "instr_count": int(len(kb.instrs)),
            "ops_count": int(len(getattr(kb, "ops", []))),
            "error": "",
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "forest_height": case.forest_height,
            "rounds": case.rounds,
            "batch_size": case.batch_size,
            "seed": case.seed,
            "cycles": -1,
            "correct": False,
            "duration_ms": elapsed_ms,
            "instr_count": -1,
            "ops_count": -1,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def run_jobs(module_path: str, cases: list[BenchCase], workers: int) -> list[dict[str, Any]]:
    jobs = [{"module_path": module_path, "case": dataclasses.asdict(c)} for c in cases]
    if workers <= 1:
        return [_run_case(job) for job in jobs]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_case, jobs))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [r for r in rows if r["correct"] and r["cycles"] >= 0]
    bad = [r for r in rows if not r["correct"] or r["cycles"] < 0]
    if good:
        cycles = [r["cycles"] for r in good]
        mean = sum(cycles) / len(cycles)
        best = min(cycles)
        worst = max(cycles)
    else:
        mean = None
        best = None
        worst = None
    return {
        "total": len(rows),
        "good": len(good),
        "bad": len(bad),
        "mean_cycles": mean,
        "best_cycles": best,
        "worst_cycles": worst,
    }


def _git_hash_for(path: str) -> str:
    module_dir = str(Path(path).resolve().parent)
    try:
        out = subprocess.check_output(
            ["git", "-C", module_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            label TEXT,
            module_path TEXT NOT NULL,
            git_hash TEXT,
            python_version TEXT NOT NULL,
            forest_height INTEGER NOT NULL,
            rounds INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            cycles INTEGER NOT NULL,
            correct INTEGER NOT NULL,
            duration_ms REAL NOT NULL,
            instr_count INTEGER NOT NULL,
            ops_count INTEGER NOT NULL,
            error TEXT NOT NULL
        )
        """
    )


def write_sqlite(
    db_path: str,
    label: str,
    module_path: str,
    rows: list[dict[str, Any]],
) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        _ensure_db(conn)
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        git_hash = _git_hash_for(module_path)
        pyver = sys.version.split()[0]
        for row in rows:
            conn.execute(
                """
                INSERT INTO benchmark_results (
                    created_at, label, module_path, git_hash, python_version,
                    forest_height, rounds, batch_size, seed,
                    cycles, correct, duration_ms, instr_count, ops_count, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    label,
                    str(Path(module_path).resolve()),
                    git_hash,
                    pyver,
                    row["forest_height"],
                    row["rounds"],
                    row["batch_size"],
                    row["seed"],
                    row["cycles"],
                    1 if row["correct"] else 0,
                    row["duration_ms"],
                    row["instr_count"],
                    row["ops_count"],
                    row["error"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark perf_takehome variants")
    parser.add_argument("--module", default="perf_takehome.py", help="Path to module under test")
    parser.add_argument(
        "--cases",
        default="10,16,256",
        help="Semicolon-separated H,R,B cases (e.g. '10,16,256;7,12,128')",
    )
    parser.add_argument("--seeds", default="123", help="Comma-separated seeds")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--label", default="manual")
    parser.add_argument("--db", default="experiments/bench_results.sqlite")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    cases = expand_cases(args.cases, args.seeds)
    rows = run_jobs(args.module, cases, args.workers)
    summary = summarize(rows)

    print(
        f"cases={summary['total']} good={summary['good']} bad={summary['bad']} "
        f"best={summary['best_cycles']} mean={summary['mean_cycles']} "
        f"worst={summary['worst_cycles']}"
    )
    for row in rows:
        status = "OK" if row["correct"] and row["cycles"] >= 0 else "FAIL"
        print(
            f"{status} H={row['forest_height']} R={row['rounds']} B={row['batch_size']} "
            f"seed={row['seed']} cycles={row['cycles']}"
        )
        if row["error"]:
            print(row["error"])

    if not args.no_db:
        write_sqlite(args.db, args.label, args.module, rows)
    if args.json_out:
        write_json(args.json_out, {"summary": summary, "rows": rows})
    if args.csv_out:
        write_csv(args.csv_out, rows)

    return 0 if summary["bad"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
