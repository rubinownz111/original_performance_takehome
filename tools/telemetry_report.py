#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _to_number(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return v
    if v is None:
        return 0
    text = str(v).strip()
    if text == "":
        return 0
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
    return [{k: _to_number(v) for k, v in row.items()} for row in rows]


def build_html(rows: list[dict[str, Any]], src_path: str) -> str:
    if not rows:
        raise ValueError("Telemetry file has no rows")

    defer_keys = sorted({k for row in rows for k in row if k.startswith("defer_")})
    defer_totals = {k: int(sum(int(row.get(k, 0)) for row in rows)) for k in defer_keys}
    slot_keys = ["slot_alu", "slot_valu", "slot_load", "slot_store", "slot_flow"]
    summary = {
        "cycles": len(rows),
        "scheduled_total": int(sum(int(row.get("scheduled", 0)) for row in rows)),
        "deferred_total": int(sum(int(row.get("deferred", 0)) for row in rows)),
        "scratch_peak": int(max(int(row.get("scratch_peak", 0)) for row in rows)),
    }

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Scheduler Telemetry Report</title>
<style>
body {{ font-family: Helvetica, Arial, sans-serif; margin: 20px; color: #1e1e1e; }}
h1, h2 {{ margin: 8px 0; }}
section {{ margin-top: 18px; }}
canvas {{ border: 1px solid #ccc; width: 100%; max-width: 1280px; height: 280px; }}
code {{ background: #f3f3f3; padding: 2px 4px; }}
table {{ border-collapse: collapse; margin-top: 8px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>Scheduler Telemetry Report</h1>
<div class=\"small\">Source: <code>{src_path}</code></div>
<section>
<h2>Summary</h2>
<ul>
<li>Cycles recorded: <b>{summary['cycles']}</b></li>
<li>Total scheduled ops: <b>{summary['scheduled_total']}</b></li>
<li>Total deferred ops: <b>{summary['deferred_total']}</b></li>
<li>Peak scratch usage: <b>{summary['scratch_peak']}</b></li>
</ul>
</section>
<section>
<h2>Engine Slot Usage by Cycle</h2>
<canvas id=\"slots\" width=\"1280\" height=\"280\"></canvas>
</section>
<section>
<h2>Queue and Scratch</h2>
<canvas id=\"state\" width=\"1280\" height=\"280\"></canvas>
</section>
<section>
<h2>Deferred Reason Totals</h2>
<table>
<tr><th>Reason</th><th>Count</th></tr>
{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in defer_totals.items())}
</table>
</section>
<script>
const rows = {json.dumps(rows)};
const slotKeys = {json.dumps(slot_keys)};
const slotColors = {{
  slot_alu: '#e4572e',
  slot_valu: '#17bebb',
  slot_load: '#ffc914',
  slot_store: '#2e282a',
  slot_flow: '#76b041',
  scheduled: '#6c5ce7',
  deferred: '#d63031',
  ready_start: '#0984e3',
  scratch_current: '#2d3436'
}};

function drawSeries(canvasId, keys) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const pad = 28;
  const w = canvas.width - 2 * pad;
  const h = canvas.height - 2 * pad;
  const n = rows.length;
  let maxY = 1;
  for (const key of keys) {{
    for (const row of rows) {{
      const v = Number(row[key] || 0);
      if (v > maxY) maxY = v;
    }}
  }}

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#888';
  ctx.strokeRect(pad, pad, w, h);

  for (const key of keys) {{
    ctx.beginPath();
    ctx.strokeStyle = slotColors[key] || '#555';
    ctx.lineWidth = 1.6;
    for (let i = 0; i < n; i++) {{
      const x = pad + (n <= 1 ? 0 : (i / (n - 1)) * w);
      const y = pad + h - (Number(rows[i][key] || 0) / maxY) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  let legendX = pad;
  const legendY = 16;
  ctx.font = '12px sans-serif';
  for (const key of keys) {{
    ctx.fillStyle = slotColors[key] || '#555';
    ctx.fillRect(legendX, legendY - 8, 12, 12);
    ctx.fillStyle = '#222';
    ctx.fillText(key, legendX + 16, legendY + 2);
    legendX += 100;
  }}
}}

drawSeries('slots', slotKeys);
drawSeries('state', ['scheduled', 'deferred', 'ready_start', 'scratch_current']);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HTML telemetry report")
    parser.add_argument("--input", required=True, help="Telemetry .csv or .jsonl")
    parser.add_argument("--output", default="artifacts/telemetry_report.html")
    args = parser.parse_args()

    src = Path(args.input)
    rows = load_rows(src)
    html = build_html(rows, str(src.resolve()))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
