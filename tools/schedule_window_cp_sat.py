#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problem import SLOT_LIMITS

try:
    from ortools.sat.python import cp_model
except Exception as exc:  # pragma: no cover
    cp_model = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


def _load_module(module_path: str):
    path = Path(module_path).resolve()
    name = f"cp_sat_mod_{path.stem}_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_ops(mod: Any, forest_height: int, rounds: int, batch_size: int):
    kb = mod.KernelBuilder()
    n_nodes = (1 << (forest_height + 1)) - 1
    with contextlib.redirect_stdout(io.StringIO()):
        kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return kb.ops


def main() -> int:
    parser = argparse.ArgumentParser(description="Find optimal schedule for a small op window via CP-SAT")
    parser.add_argument("--module", default="perf_takehome.py")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if cp_model is None:
        print(f"ortools import failed: {IMPORT_ERROR}")
        return 2

    mod = _load_module(args.module)
    ops = _build_ops(mod, args.forest_height, args.rounds, args.batch_size)

    start = args.window_start
    end = min(len(ops), start + args.window_size)
    if start >= end:
        raise SystemExit("Empty window")
    window = ops[start:end]

    local_index_by_id = {op.id: i for i, op in enumerate(window)}
    preds_local: list[list[int]] = [[] for _ in window]
    for i, op in enumerate(window):
        for dep_id in op.deps:
            j = local_index_by_id.get(dep_id)
            if j is not None:
                preds_local[i].append(j)

    model = cp_model.CpModel()
    horizon = args.horizon

    starts = [model.NewIntVar(0, horizon - 1, f"s_{i}") for i in range(len(window))]
    x: list[list[cp_model.IntVar]] = []
    for i in range(len(window)):
        row = [model.NewBoolVar(f"x_{i}_{t}") for t in range(horizon)]
        x.append(row)
        model.Add(sum(row) == 1)
        model.Add(starts[i] == sum(t * row[t] for t in range(horizon)))

    for i in range(len(window)):
        for j in preds_local[i]:
            model.Add(starts[i] >= starts[j] + 1)

    engine_to_ops: dict[str, list[int]] = {}
    for i, op in enumerate(window):
        engine_to_ops.setdefault(op.engine, []).append(i)

    for engine, indices in engine_to_ops.items():
        limit = SLOT_LIMITS.get(engine)
        if limit is None:
            continue
        for t in range(horizon):
            model.Add(sum(x[i][t] for i in indices) <= limit)

    makespan = model.NewIntVar(1, horizon, "makespan")
    for s in starts:
        model.Add(makespan >= s + 1)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.timeout_sec
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    print(f"status={status_name}")
    payload: dict[str, Any] = {
        "status": status_name,
        "window_start": start,
        "window_end": end,
        "window_size": len(window),
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        best = int(solver.Value(makespan))
        payload["makespan"] = best
        schedule: list[dict[str, Any]] = []
        for i, op in enumerate(window):
            schedule.append(
                {
                    "local_idx": i,
                    "global_idx": start + i,
                    "engine": op.engine,
                    "time": int(solver.Value(starts[i])),
                }
            )
        schedule.sort(key=lambda r: (r["time"], r["global_idx"]))
        payload["schedule"] = schedule
        print(f"optimal_makespan={best}")
    else:
        print("No feasible schedule found in search horizon")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0 if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
