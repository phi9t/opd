import subprocess
import sys
from pathlib import Path

import yaml


def _write_minimal_config(path: Path, run_dir: Path, max_steps: int = 1) -> None:
    base = yaml.safe_load(Path("configs/tier_tiny.yaml").read_text())
    base["max_steps"] = max_steps
    base["run_dir"] = str(run_dir)
    path.write_text(yaml.dump(base))


def test_train_cli_two_steps(tmp_path):
    config_path = tmp_path / "tier_tiny.yaml"
    _write_minimal_config(config_path, tmp_path, max_steps=2)

    subprocess.run(
        [sys.executable, "-m", "opd.cli", "train", "--config", str(config_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    runs = sorted(tmp_path.glob("run_*"))
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "steps" / "0.json").exists()
    assert (run_dir / "steps" / "1.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "config.yaml").exists()
