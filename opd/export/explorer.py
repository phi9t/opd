from __future__ import annotations

import json
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _REPO_ROOT / "explorer" / "public" / "data" / "index.json"

_GLOSSARY = {
    "opd": "Rollout + teacher logprob service + KL update",
    "ppo": "Proximal Policy Optimization — clipped policy-gradient RL with a value baseline",
    "grpo": "Group Relative Policy Optimization — advantages normalized within prompt groups",
}


def _load_config(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def _load_steps(run_dir: Path) -> list[dict]:
    steps_dir = run_dir / "steps"
    paths = sorted(steps_dir.glob("*.json"), key=lambda p: int(p.stem))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _index_rel_path(out_path: Path) -> str:
    data_dir = _REPO_ROOT / "explorer" / "public" / "data"
    try:
        return out_path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return f"runs/{out_path.name}"


def _update_index(run_id: str, out_path: Path, tier: str) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _INDEX_PATH.exists():
        index = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    else:
        index = []

    entry = {
        "id": run_id,
        "path": _index_rel_path(out_path),
        "tier": tier,
    }
    index = [e for e in index if e.get("id") != run_id]
    index.append(entry)
    index.sort(key=lambda e: e.get("id", ""))
    _INDEX_PATH.write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )


def export_run(run_dir: Path, out_path: Path) -> None:
    """Export a training run directory to an explorer JSON bundle."""
    run_dir = run_dir.resolve()
    cfg = _load_config(run_dir)
    steps = _load_steps(run_dir)

    bundle = {
        "run_id": run_dir.name,
        "tier": cfg["tier"],
        "runtime": cfg["runtime"],
        "device": cfg["device"],
        "loss_mode": cfg["loss_mode"],
        "teacher_signal": cfg["teacher_signal"],
        "models": {
            "student_hidden_size": cfg["student_hidden_size"],
            "teacher_hidden_size": cfg["teacher_hidden_size"],
            "vocab_size": cfg["vocab_size"],
        },
        "steps": steps,
        "glossary": _GLOSSARY,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    _update_index(run_dir.name, out_path, cfg["tier"])
