# Performance Tooling

This folder contains local tooling for faster optimization iteration.

## 1) Scheduler Telemetry

Enable per-cycle telemetry by setting `SCHED_TELEMETRY_PATH`:

```bash
SCHED_TELEMETRY_PATH=artifacts/scheduler_telemetry.csv python tests/submission_tests.py
```

Supported formats:
- `.csv` (recommended)
- `.jsonl`

## 2) Benchmark Runner (A single variant)

Runs correctness + cycle checks across multiple cases/seeds, with optional parallel workers.

```bash
python tools/bench_suite.py \
  --module perf_takehome.py \
  --cases "10,16,256;7,12,128" \
  --seeds "1,2,3,42,123" \
  --workers 8 \
  --label main
```

Writes results to SQLite by default: `experiments/bench_results.sqlite`.

## 3) A/B Runner

```bash
python tools/bench_ab.py \
  --base perf_takehome.py \
  --candidate perf_takehome_candidate.py \
  --cases "10,16,256" \
  --seeds "1,2,3,42,123" \
  --workers 8
```

## 4) Parallel Experiment Runner

Run many module variants concurrently.

```bash
python tools/parallel_experiments.py \
  --experiments-file experiments/sample_experiments.json \
  --cases "10,16,256" \
  --seeds "1,2,3,42,123" \
  --workers-per-exp 4 \
  --max-parallel 4
```

## 5) CP-SAT Window Solver (OR-Tools)

Searches for an optimal schedule on a small op window.

```bash
.venv/bin/python tools/schedule_window_cp_sat.py \
  --module perf_takehome.py \
  --window-start 0 \
  --window-size 40 \
  --horizon 80 \
  --timeout-sec 20 \
  --out artifacts/window_solution.json
```

## 6) Telemetry HTML Report

```bash
python tools/telemetry_report.py \
  --input artifacts/scheduler_telemetry.csv \
  --output artifacts/telemetry_report.html
```

Open the generated HTML in a browser for slot-usage, queue/scratch, and defer reason trends.
