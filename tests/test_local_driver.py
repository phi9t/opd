import json
from dataclasses import replace
from pathlib import Path

from opd.config import load_config
from opd.runtime.local_driver import LocalDriver


def _find_run_dir(base: Path) -> Path:
    if (base / "steps").is_dir():
        return base
    runs = sorted(base.glob("run_*"))
    assert len(runs) == 1, f"expected one run_* dir under {base}, got {runs}"
    return runs[0]


def test_local_driver_two_steps(tmp_path):
    cfg = load_config(Path("configs/tier_tiny.yaml"))
    cfg = replace(cfg, run_dir=str(tmp_path), max_steps=2, batch_size=2)
    driver = LocalDriver(cfg)
    driver.run()

    run_dir = _find_run_dir(tmp_path)
    assert (run_dir / "steps" / "0.json").exists()
    assert (run_dir / "steps" / "1.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.yaml").exists()

    step0 = json.loads((run_dir / "steps" / "0.json").read_text())
    assert "samples" in step0
    assert len(step0["samples"]) > 0
    sample = step0["samples"][0]
    assert isinstance(sample["tokens"], list)
    assert len(sample["tokens"]) == len(sample["kl"])
    assert all(isinstance(x, float) for x in sample["kl"])


def test_local_driver_disables_token_samples(tmp_path):
    cfg = load_config(Path("configs/tier_tiny.yaml"))
    cfg = replace(
        cfg, run_dir=str(tmp_path), max_steps=1, batch_size=2, log_token_samples=0
    )
    driver = LocalDriver(cfg)
    driver.run()

    run_dir = _find_run_dir(tmp_path)
    step0 = json.loads((run_dir / "steps" / "0.json").read_text())
    assert step0["samples"] == []
