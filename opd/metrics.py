from __future__ import annotations

import json
from pathlib import Path


def write_step_json(run_dir: Path, step: int, payload: dict) -> None:
    steps_dir = run_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    path = steps_dir / f"{step}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_metrics_jsonl(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
