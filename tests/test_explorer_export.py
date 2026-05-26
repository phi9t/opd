import json
from dataclasses import replace
from pathlib import Path

from opd.config import load_config
from opd.export.explorer import export_run
from opd.runtime.local_driver import LocalDriver


def _find_run_dir(base: Path) -> Path:
    if (base / "steps").is_dir():
        return base
    runs = sorted(base.glob("run_*"))
    assert len(runs) == 1, f"expected one run_* dir under {base}, got {runs}"
    return runs[0]


def test_export_run_produces_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "opd.export.explorer._INDEX_PATH",
        tmp_path / "index.json",
    )

    cfg = load_config(Path("configs/tier_tiny.yaml"))
    cfg = replace(cfg, run_dir=str(tmp_path), max_steps=2, batch_size=2)
    LocalDriver(cfg).run()

    run_dir = _find_run_dir(tmp_path)
    out_path = tmp_path / "bundle.json"
    export_run(run_dir, out_path)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "steps" in data
    assert len(data["steps"]) == 2
    assert data["run_id"] == run_dir.name
    assert data["tier"] == "tiny"
    assert "glossary" in data
    assert "opd" in data["glossary"]
