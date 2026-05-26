# OPD Compute Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a hackable OPD trainer where the `tiny` tier runs a full on-policy distillation loop on CPU (synthetic Qwen3-style models + tiny-stories), exports metrics to a clinique-style explorer UI, then extend to `qwen3_small` on GPU (M2+).

**Architecture:** Shared batch types and loss modules feed a `runtime=local` CPU driver (M1) and a `runtime=ray` GPU driver (M2+). `tiny` uses eager PyTorch rollout/teacher/actor modules; `qwen3_small` swaps in vLLM + FSDP. Python exports static JSON; React explorer renders it.

**Tech Stack:** Python 3.11+, PyTorch 2.x, PyYAML, pytest, Vite, React 19, TypeScript, optional Ray/vLLM (M2+ only).

**Spec:** `docs/superpowers/specs/2026-05-26-opd-compute-infra-design.md`

---

## Milestone Map

| Milestone | This plan section | Deliverable |
|-----------|-------------------|-------------|
| **M1** | Tasks 1–16 | CPU `tiny` E2E + explorer + tutorial 01 |
| **M2** | Tasks 17–20 | `qwen3_small` Ray + vLLM + FSDP + top-k |
| **M3** | Tasks 21–22 | `opd_rl` loss + tutorial 03 |
| **M4** | Task 23 | Multi-node placement groups |

Execute **M1 first**; each milestone should leave the repo in a working, testable state.

---

## File Structure (M1)

```text
pyproject.toml
README.md
configs/tier_tiny.yaml
opd/
  __init__.py
  cli.py
  config.py
  batches.py
  metrics.py
  models/
    __init__.py
    synthetic_qwen3.py
  loss/
    __init__.py
    kl.py
  data/
    __init__.py
    prompts.py
    tokenizer_tiny.py
  rollout/
    __init__.py
    eager.py
  teacher/
    __init__.py
    eager.py
  actor/
    __init__.py
    eager.py
  runtime/
    __init__.py
    local_driver.py
  export/
    __init__.py
    explorer.py
scripts/
  gen_tiny_stories.py
  run_lab.sh
data/
  prompts/tiny_stories.jsonl      # committed
  tokenizer/tiny/vocab.json       # committed
tests/
  test_batches.py
  test_kl_loss.py
  test_synthetic_qwen3.py
  test_local_driver.py
  test_explorer_export.py
explorer/
  package.json
  src/App.tsx
  src/opd/...
docs/tutorials/01_three_pools.md
```

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `opd/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add `pyproject.toml`**

```toml
[project]
name = "opd"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
ray = ["ray[default]>=2.9"]
vllm = ["vllm>=0.6"]  # M2+ only

[project.scripts]
opd = "opd.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["opd"]
```

- [ ] **Step 2: Install and verify**

```bash
cd /Users/phi9t/CodeBase/opd
python -m pip install -e ".[dev]"
pytest --collect-only
```

Expected: collects 0 tests (no failures).

- [ ] **Step 3: Commit**

```bash
git init
git add pyproject.toml README.md opd/__init__.py tests/conftest.py
git commit -m "chore: scaffold opd Python package"
```

---

## Task 2: Configuration loading

**Files:**
- Create: `opd/config.py`
- Create: `configs/tier_tiny.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing config test**

```python
# tests/test_config.py
from pathlib import Path
from opd.config import load_config

def test_load_tier_tiny():
    cfg = load_config(Path("configs/tier_tiny.yaml"))
    assert cfg.tier == "tiny"
    assert cfg.runtime == "local"
    assert cfg.device == "cpu"
    assert cfg.engine == "eager"
    assert cfg.loss_mode == "kl"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/test_config.py::test_load_tier_tiny -v
```

- [ ] **Step 3: Implement `opd/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class TrainConfig:
    tier: str
    runtime: str
    device: str
    engine: str
    loss_mode: str
    teacher_signal: str
    max_steps: int
    batch_size: int
    max_new_tokens: int
    lr: float
    seed: int
    run_dir: str
    student_hidden_size: int
    teacher_hidden_size: int
    vocab_size: int
    prompts_path: str
    tokenizer_dir: str

def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text())
    return TrainConfig(
        tier=raw["tier"],
        runtime=raw["runtime"],
        device=raw["device"],
        engine=raw["engine"],
        loss_mode=raw.get("loss", {}).get("mode", "kl"),
        teacher_signal=raw.get("teacher", {}).get("signal", "sampled"),
        max_steps=int(raw.get("max_steps", 20)),
        batch_size=int(raw.get("batch_size", 4)),
        max_new_tokens=int(raw.get("max_new_tokens", 32)),
        lr=float(raw.get("lr", 1e-4)),
        seed=int(raw.get("seed", 42)),
        run_dir=raw.get("run_dir", "runs"),
        student_hidden_size=int(raw["model"]["student_hidden_size"]),
        teacher_hidden_size=int(raw["model"]["teacher_hidden_size"]),
        vocab_size=int(raw["model"]["vocab_size"]),
        prompts_path=raw["data"]["prompts_path"],
        tokenizer_dir=raw["data"]["tokenizer_dir"],
    )

def validate_config(cfg: TrainConfig) -> None:
    if cfg.tier == "tiny" and cfg.runtime != "local":
        raise ValueError("tiny tier requires runtime=local in v1")
    if cfg.tier == "tiny" and cfg.device != "cpu":
        raise ValueError("tiny tier requires device=cpu in v1")
    if cfg.loss_mode == "opd_rl" and cfg.tier == "tiny":
        raise ValueError("opd_rl is M3; use qwen3_small tier")
```

- [ ] **Step 4: Add `configs/tier_tiny.yaml`**

```yaml
tier: tiny
runtime: local
device: cpu
engine: eager
max_steps: 20
batch_size: 4
max_new_tokens: 32
lr: 0.0001
seed: 42
run_dir: runs

model:
  student_hidden_size: 256
  teacher_hidden_size: 512
  vocab_size: 8192

loss:
  mode: kl

teacher:
  signal: sampled

data:
  prompts_path: data/prompts/tiny_stories.jsonl
  tokenizer_dir: data/tokenizer/tiny
```

- [ ] **Step 5: Run test — expect PASS**

```bash
pytest tests/test_config.py -v
```

- [ ] **Step 6: Commit**

```bash
git add opd/config.py configs/tier_tiny.yaml tests/test_config.py
git commit -m "feat: add tier config loader and tiny defaults"
```

---

## Task 3: Batch types

**Files:**
- Create: `opd/batches.py`
- Create: `tests/test_batches.py`

- [ ] **Step 1: Write failing serde test**

```python
import torch
from opd.batches import TrajectoryBatch, TeacherBatch, TrainBatch

def test_trajectory_roundtrip_dict():
    traj = TrajectoryBatch(
        prompt_ids=torch.tensor([[1, 2]]),
        token_ids=torch.tensor([[3, 4]]),
        student_logprobs=torch.tensor([[-1.0, -1.1]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        meta={"step": 0},
    )
    d = traj.to_dict()
    traj2 = TrajectoryBatch.from_dict(d)
    assert torch.equal(traj2.token_ids, traj.token_ids)
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `opd/batches.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
import torch

def _t(x):  # tensor ↔ list for JSON
    return x.detach().cpu().tolist()

def _lt(x):
    return torch.tensor(x)

@dataclass
class TrajectoryBatch:
    prompt_ids: torch.Tensor
    token_ids: torch.Tensor
    student_logprobs: torch.Tensor
    attention_mask: torch.Tensor
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_ids": _t(self.prompt_ids),
            "token_ids": _t(self.token_ids),
            "student_logprobs": _t(self.student_logprobs),
            "attention_mask": _t(self.attention_mask),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryBatch:
        return cls(
            prompt_ids=_lt(d["prompt_ids"]),
            token_ids=_lt(d["token_ids"]),
            student_logprobs=_lt(d["student_logprobs"]),
            attention_mask=_lt(d["attention_mask"]),
            meta=d.get("meta", {}),
        )

@dataclass
class TeacherBatch:
    teacher_logprobs: torch.Tensor
    topk_ids: torch.Tensor | None = None
    topk_logprobs: torch.Tensor | None = None

@dataclass
class TrainBatch:
    trajectory: TrajectoryBatch
    teacher: TeacherBatch
    old_logprobs: torch.Tensor | None = None
```

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

```bash
git add opd/batches.py tests/test_batches.py
git commit -m "feat: add TrajectoryBatch and TeacherBatch types"
```

---

## Task 4: Reverse KL loss

**Files:**
- Create: `opd/loss/kl.py`
- Create: `tests/test_kl_loss.py`

- [ ] **Step 1: Write failing KL test**

```python
import torch
from opd.loss.kl import reverse_kl_loss

def test_reverse_kl_known_values():
    student = torch.tensor([-1.0, -2.0])
    teacher = torch.tensor([-1.5, -1.0])
    loss = reverse_kl_loss(student, teacher)
    # mean of (log s - log t)
    expected = torch.tensor([-1.0 - (-1.5), -2.0 - (-1.0)]).mean()
    assert torch.allclose(loss, expected)
```

- [ ] **Step 2: Implement `opd/loss/kl.py`**

```python
import torch

def reverse_kl_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    per_token = student_logprobs - teacher_logprobs
    if mask is not None:
        per_token = per_token * mask
        return per_token.sum() / mask.sum().clamp(min=1)
    return per_token.mean()
```

- [ ] **Step 3: Run — PASS; commit**

```bash
git add opd/loss/ tests/test_kl_loss.py
git commit -m "feat: add sampled-token reverse KL loss"
```

---

## Task 5: Synthetic Qwen3-style model

**Files:**
- Create: `opd/models/synthetic_qwen3.py`
- Create: `tests/test_synthetic_qwen3.py`

- [ ] **Step 1: Write failing forward-shape test**

```python
import torch
from opd.models.synthetic_qwen3 import SyntheticQwen3, SyntheticQwen3Config

def test_forward_logits_shape():
    cfg = SyntheticQwen3Config(vocab_size=128, hidden_size=64, num_hidden_layers=2)
    m = SyntheticQwen3(cfg)
    x = torch.randint(0, 128, (2, 8))
    logits = m(x).logits
    assert logits.shape == (2, 8, 128)
```

- [ ] **Step 2: Implement minimal model** (`opd/models/synthetic_qwen3.py`)

Implement:
- `SyntheticQwen3Config` dataclass: `vocab_size`, `hidden_size`, `num_hidden_layers=2`, `num_attention_heads=4`, `num_key_value_heads=2`, `max_seq_len=512`
- `RMSNorm`, `RotaryEmbedding` (basic), `GQAAttention`, `SwiGLUMLP`, `DecoderBlock`, `SyntheticQwen3ForCausalLM` returning `ModelOutput(logits=...)`
- `logprobs_on_tokens(input_ids, target_ids)` helper: forward full sequence, gather log-softmax at continuation positions

Keep each class in the same file for M1 (split later if >400 lines).

- [ ] **Step 3: Run shape + logprob test**

```python
def test_logprobs_on_sampled_tokens():
    cfg = SyntheticQwen3Config(vocab_size=64, hidden_size=32, num_hidden_layers=2)
    m = SyntheticQwen3(cfg)
    prompt = torch.tensor([[1, 2, 3]])
    cont = torch.tensor([[4, 5]])
    full = torch.cat([prompt, cont], dim=1)
    lp = m.logprobs_on_tokens(full, cont)
    assert lp.shape == (1, 2)
```

- [ ] **Step 4: Commit**

```bash
git add opd/models/ tests/test_synthetic_qwen3.py
git commit -m "feat: add synthetic Qwen3-style micro LM"
```

---

## Task 6: Tiny tokenizer + story data

**Files:**
- Create: `opd/data/tokenizer_tiny.py`
- Create: `opd/data/prompts.py`
- Create: `scripts/gen_tiny_stories.py`
- Create: `data/tokenizer/tiny/vocab.json`
- Create: `data/prompts/tiny_stories.jsonl`

- [ ] **Step 1: Implement word-level tokenizer** (`opd/data/tokenizer_tiny.py`)

- Load `vocab.json` mapping token str → id; unk=0, pad=1.
- `encode(text) -> list[int]`, `decode(ids) -> str`
- Build vocab from stories in `gen_tiny_stories.py` (max vocab 8192)

- [ ] **Step 2: Implement `scripts/gen_tiny_stories.py`**

Generate 200 deterministic story prompts (seeded template expansion), write:
- `data/prompts/tiny_stories.jsonl` — one JSON object per line: `{"prompt": "..."}`
- `data/tokenizer/tiny/vocab.json`

- [ ] **Step 3: Run generator and commit data**

```bash
python scripts/gen_tiny_stories.py
```

- [ ] **Step 4: `opd/data/prompts.py`**

```python
def load_prompts(path: str, limit: int | None = None) -> list[str]:
    ...
```

- [ ] **Step 5: Commit**

```bash
git add opd/data/ scripts/gen_tiny_stories.py data/
git commit -m "feat: add tiny story prompts and tokenizer"
```

---

## Task 7: Eager rollout + teacher + actor modules

**Files:**
- Create: `opd/rollout/eager.py`
- Create: `opd/teacher/eager.py`
- Create: `opd/actor/eager.py`

- [ ] **Step 1: `StudentRollout` (`opd/rollout/eager.py`)**

- Holds `SyntheticQwen3` student in eval mode
- `generate(prompt_ids: torch.Tensor) -> TrajectoryBatch`:
  - Greedy/sample loop up to `max_new_tokens`
  - Record `student_logprobs` at each generated token
  - Build `attention_mask` over prompt+continuation

- [ ] **Step 2: `TeacherScorer` (`opd/teacher/eager.py`)**

- Holds wider frozen teacher (`hidden_size` from config)
- `score(traj) -> TeacherBatch`:
  - Concatenate prompt+continuation per row
  - `teacher_logprobs = teacher.logprobs_on_tokens(full, continuation)`

- [ ] **Step 3: `StudentActor` (`opd/actor/eager.py`)**

- Train mode student + AdamW
- `train_step(batch: TrainBatch) -> dict`:
  - Recompute `student_logprobs` on continuation tokens
  - `loss = reverse_kl_loss(student_lp, teacher_lp, mask)`
  - `backward()`, `step()`, return `{loss, grad_norm, mean_kl}`

- [ ] **Step 4: Manual smoke in REPL** (optional); commit

```bash
git add opd/rollout/ opd/teacher/ opd/actor/
git commit -m "feat: add eager rollout, teacher, and actor modules"
```

---

## Task 8: Metrics + local driver

**Files:**
- Create: `opd/metrics.py`
- Create: `opd/runtime/local_driver.py`
- Create: `tests/test_local_driver.py`

- [ ] **Step 1: `opd/metrics.py`**

```python
def write_step_json(run_dir: Path, step: int, payload: dict) -> None: ...
def append_metrics_jsonl(run_dir: Path, payload: dict) -> None: ...
```

Payload must include: `gen_ms`, `teacher_ms`, `train_ms`, `sync_ms`, `sync_bytes`, `mean_kl`, `loss`, `grad_norm`, `loss_mode`, `teacher_signal`.

- [ ] **Step 2: Write failing 2-step integration test**

```python
from dataclasses import replace
from pathlib import Path
from opd.config import load_config
from opd.runtime.local_driver import LocalDriver

def test_local_driver_two_steps(tmp_path):
    cfg = load_config(Path("configs/tier_tiny.yaml"))
    cfg = replace(cfg, run_dir=str(tmp_path), max_steps=2)
    driver = LocalDriver(cfg)
    driver.run()
    assert (tmp_path / "steps" / "0.json").exists()
    assert (tmp_path / "steps" / "1.json").exists()
```

- [ ] **Step 3: Implement `LocalDriver`**

```python
class LocalDriver:
    def __init__(self, cfg: TrainConfig): ...

    def run(self) -> None:
        torch.manual_seed(cfg.seed)
        tokenizer = TinyTokenizer(cfg.tokenizer_dir)
        prompts = load_prompts(cfg.prompts_path)
        student_rollout = StudentRollout(...)
        teacher = TeacherScorer(...)
        actor = StudentActor(...)
        for step in range(cfg.max_steps):
            # time gen → teacher → train → sync
            # after train: rollout.load_state_dict(actor.state_dict())
            # write step json + metrics.jsonl
```

- [ ] **Step 4: Run integration test — PASS** (target <30s CPU)

```bash
pytest tests/test_local_driver.py -v
```

- [ ] **Step 5: Commit**

```bash
git add opd/metrics.py opd/runtime/ tests/test_local_driver.py
git commit -m "feat: add CPU local driver and step metrics"
```

---

## Task 9: CLI entrypoint

**Files:**
- Create: `opd/cli.py`
- Create: `scripts/run_lab.sh`

- [ ] **Step 1: Implement CLI**

```python
# opd/cli.py
import argparse
from pathlib import Path
from opd.config import load_config, validate_config
from opd.runtime.local_driver import LocalDriver

def main() -> None:
    parser = argparse.ArgumentParser(prog="opd")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    exp = sub.add_parser("export-explorer")
    exp.add_argument("--run", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "train":
        cfg = load_config(args.config)
        validate_config(cfg)
        if cfg.runtime == "local":
            LocalDriver(cfg).run()
        else:
            raise NotImplementedError("ray runtime is M2")
    elif args.cmd == "export-explorer":
        from opd.export.explorer import export_run
        export_run(args.run, args.out)
```

- [ ] **Step 2: `scripts/run_lab.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
opd train --config configs/tier_tiny.yaml
```

- [ ] **Step 3: Run lab script**

```bash
chmod +x scripts/run_lab.sh
./scripts/run_lab.sh
ls runs/
```

- [ ] **Step 4: Commit**

```bash
git add opd/cli.py scripts/run_lab.sh
git commit -m "feat: add opd train CLI and lab script"
```

---

## Task 10: Explorer export (Python)

**Files:**
- Create: `opd/export/explorer.py`
- Create: `tests/test_explorer_export.py`

- [ ] **Step 1: Write failing export test**

```python
from opd.export.explorer import export_run

def test_export_run_produces_bundle(tmp_path):
    # use fixture run dir from test_local_driver
    export_run(run_dir, out / "demo.json")
    data = json.loads((out / "demo.json").read_text())
    assert "steps" in data
    assert "glossary" in data
```

- [ ] **Step 2: Implement `export_run`**

Build JSON:
```python
{
  "run_id": "...",
  "tier": "tiny",
  "runtime": "local",
  "device": "cpu",
  "loss_mode": "kl",
  "models": {"student_hidden": 256, "teacher_hidden": 512},
  "steps": [ ... merged step json ... ],
  "glossary": { "opd": "...", "ppo": "...", "grpo": "..." }
}
```

Also update `explorer/public/data/index.json` with run entry.

- [ ] **Step 3: Commit**

```bash
git add opd/export/ tests/test_explorer_export.py
git commit -m "feat: add explorer JSON exporter"
```

---

## Task 11: Explorer UI (Vite/React)

**Files:**
- Create: `explorer/package.json`, `explorer/vite.config.ts`, `explorer/index.html`
- Create: `explorer/src/App.tsx`, `explorer/src/opd/OpdExplorer.tsx`, etc.
- Copy/adapt CSS tokens from `clinique/explorer/src/index.css` (observatory-bg, explorer-container)

- [ ] **Step 1: Scaffold Vite app**

```bash
cd explorer && npm create vite@latest . -- --template react-ts
```

- [ ] **Step 2: Implement panels**

- `OpdExplorer.tsx`: fetch `public/data/index.json`, load selected run JSON
- `RunOverview.tsx`: tier, steps, final KL
- `StepTimeline.tsx`: stacked bar chart (gen/teacher/train/sync ms) — use recharts like clinique
- `LearningCurves.tsx`: KL + loss vs step
- `Glossary.tsx`: static markdown table from bundle

- [ ] **Step 3: Commit demo bundle**

Run tiny lab → export → save to `explorer/public/data/runs/tiny-demo.json` and `index.json`.

- [ ] **Step 4: Commit explorer**

```bash
git add explorer/
git commit -m "feat: add OPD metrics explorer UI"
```

---

## Task 12: Playwright + CI smoke

**Files:**
- Create: `explorer/playwright.config.ts`
- Create: `explorer/tests/opd.spec.ts`

- [ ] **Step 1: Add Playwright test**

Assert: page loads, run selector shows `tiny-demo`, step timeline renders 20 bars.

- [ ] **Step 2: Run**

```bash
cd explorer && npm run test:e2e
```

- [ ] **Step 3: Commit**

---

## Task 13: Tutorial 01 + README

**Files:**
- Create: `docs/tutorials/01_three_pools.md`
- Modify: `README.md`

- [ ] **Step 1: Write tutorial**

Explain:
- OPD = rollout + teacher + train + sync
- On CPU `tiny`, three modules are logical pools in one process
- How to run `./scripts/run_lab.sh` and open explorer
- Map step JSON fields to MLSys bottlenecks

- [ ] **Step 2: README quickstart**

```bash
pip install -e ".[dev]"
./scripts/run_lab.sh
opd export-explorer --run runs/<id> --out explorer/public/data/runs/<id>.json
cd explorer && npm install && npm run dev
```

- [ ] **Step 3: Commit — M1 complete**

```bash
git commit -m "docs: add tutorial 01 and README quickstart"
```

---

# M2: `qwen3_small` GPU tier (Tasks 17–20)

### Task 17: Ray placement + ray driver stub

**Files:**
- Create: `opd/runtime/placement.py`
- Create: `opd/runtime/ray_driver.py`
- Create: `configs/tier_qwen3_small.yaml`

- [ ] Implement `RayDriver` with sync loop calling remote actors
- [ ] Validate `device=cuda` and CUDA availability
- [ ] Unit test with Ray local mode mocked or skipped if no GPU

### Task 18: vLLM rollout + teacher workers

**Files:**
- Create: `opd/rollout/vllm.py`
- Create: `opd/teacher/vllm.py`

- [ ] vLLM engine for student sampling and frozen teacher logprobs
- [ ] Support `teacher.signal=topk` (k=128 default)

### Task 19: FSDP actor + weight sync

**Files:**
- Create: `opd/actor/fsdp.py`
- Create: `opd/actor/weight_sync.py`

- [ ] FSDP training step on HF Qwen3 student
- [ ] Bulk weight sync to vLLM rollout engine; log `sync_ms` / `sync_bytes`

### Task 20: HF data loader + M2 E2E

**Files:**
- Create: `opd/data/hf.py`
- Create: `docs/tutorials/02_sync_step_trace.md`

- [ ] `tier_qwen3_small.yaml` with model ids and HF dataset
- [ ] Manual/nightly: 1-step 8-GPU run documented in README

---

# M3: `opd_rl` loss (Tasks 21–22)

### Task 21: `opd/loss/opd_rl.py`

- [ ] Clipped IS loss using `old_logprobs` from rollout
- [ ] Tests with synthetic tensors
- [ ] Wire into actor train when `loss.mode=opd_rl`

### Task 22: Tutorial 03 + explorer OPD-RL panel

- [ ] `docs/tutorials/03_kl_vs_opd_rl.md`
- [ ] Explorer panel: clip fraction, ratio histogram when `loss_mode=opd_rl`

---

# M4: Multi-node (Task 23)

### Task 23: Cross-node placement groups

- [ ] Extend `placement.py` for multi-node GPU bundles
- [ ] Document in README; no change to batch types or exporter schema

---

## Spec Coverage Checklist

| Spec requirement | Task |
|------------------|------|
| CPU tiny local driver | 8 |
| Synthetic Qwen3 2-layer | 5 |
| tiny_stories + tokenizer | 6 |
| Sampled-token KL default | 4, 7 |
| Step metrics JSON | 8 |
| export-explorer | 10 |
| Explorer UI panels | 11 |
| Config tiers | 2 |
| opd CLI | 9 |
| M1 CI 2-step CPU test | 8 |
| qwen3_small vLLM+Ray | 17–20 |
| top-k teacher | 18 |
| opd_rl | 21–22 |
| Multi-node | 23 |

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-26-opd-compute-infra.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — run tasks in this session with executing-plans checkpoints  

**Which approach do you want?**
