# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

OPD ("on-policy distillation") is a research/teaching trainer that distills a teacher LM into a student via on-policy rollouts. The project is intentionally tiered: the `tiny` tier runs end-to-end on CPU with synthetic models, and larger tiers (Ray runtime, vLLM engine, RL-style losses) are planned milestones (M2/M3). See `docs/superpowers/specs/` for design docs and `docs/tutorials/01_three_pools.md` for the conceptual walkthrough.

## Commands

Python (run from repo root, using the project venv):

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/opd train --config configs/tier_tiny.yaml          # full training run
./scripts/run_lab.sh                                         # same, via wrapper
.venv/bin/pytest                                             # all tests
.venv/bin/pytest tests/test_local_driver.py::test_name -x    # single test
.venv/bin/opd export-explorer --run runs/<run_dir> --out explorer/public/data/runs/<name>.json
```

Explorer (React + Vite, from `explorer/`):

```bash
npm install
npm run dev          # local dev server
npm run build        # tsc -b && vite build
npm run lint         # eslint
npm run test:e2e     # Playwright smoke
```

There is no Python linter/formatter configured; do not invent one.

## Architecture

### The three "pools" (per training step)

`opd/runtime/local_driver.py` is the orchestrator. Each step runs three pools sequentially and times each:

1. **Rollout** (`opd/rollout/eager.py` → `StudentRollout`): student generates trajectories from prompts.
2. **Teacher** (`opd/teacher/eager.py` → `TeacherScorer`): scores trajectories, returns `TeacherBatch` (full logprobs and/or top-k).
3. **Actor** (`opd/actor/eager.py` → `StudentActor`): one gradient step using the configured loss; returns metrics.

After the actor step, `LocalDriver` copies the actor's `state_dict` into the rollout model — this weight sync is what makes the next rollout "on-policy" with respect to the updated student. The size of that sync (`sync_bytes`) is recorded as a first-class metric because it is the dominant cost at larger tiers.

### Data flow contracts

`opd/batches.py` defines the three tensor containers that flow between pools — keep these stable:

- `TrajectoryBatch` (rollout → teacher → actor): `prompt_ids`, `token_ids`, `student_logprobs`, `attention_mask`. Has `to_dict` / `from_dict` for serialization.
- `TeacherBatch` (teacher → actor): `teacher_logprobs` plus optional top-k.
- `TrainBatch` (actor input): wraps the above plus optional `old_logprobs` (for future PPO-style losses).

### Configuration & tier gating

`opd/config.py` flattens the YAML into a frozen `TrainConfig` dataclass and enforces tier invariants in `validate_config`:

- `tier: tiny` ⇒ `runtime=local` and `device=cpu` (anything else raises).
- `loss.mode: opd_rl` is rejected on the `tiny` tier (it is an M3 feature).

When adding new tiers/runtimes/engines, extend `validate_config` rather than scattering checks.

### Models

`opd/models/synthetic_qwen3.py` is a deliberately tiny stand-in for a real LM — it lets the whole pipeline run on CPU in tests. Real model integration is a later milestone; do not add HF/transformers imports to the tiny path.

### Run artifacts

Every `LocalDriver.run()` writes to `runs/run_<utc-timestamp>/`:

- `config.yaml` — snapshot of the resolved `TrainConfig`.
- `steps/<step>.json` — per-step payload (timings, loss, grad_norm, mean_kl, sync_bytes, …).
- `metrics.jsonl` — same payloads appended one per line for streaming consumers.

`opd/export/explorer.py` (`export_run`) consolidates a run dir into a single JSON blob the React explorer reads. The explorer is a pure consumer — it never imports Python and only reads files under `explorer/public/data/runs/`.

### Explorer (React)

`explorer/src/opd/` contains the OPD-specific views (`OpdExplorer`, `RunOverview`, `LearningCurves`, `StepTimeline`, `Glossary`) plus the `types.ts` shape that mirrors the exporter output. The `runs/` directory is gitignored except for `explorer/public/data/runs/**` (see `.gitignore`) so exported runs can be committed for demos.

## Conventions worth knowing

- The CLI entry point is `opd.cli:main`; subcommands are wired with `argparse` — keep new commands there.
- Python target is 3.11+ and uses `from __future__ import annotations` consistently.
- Heavy/optional deps are in extras: `[dev]` (pytest), `[ray]`, `[vllm]`. Don't add them to the base `dependencies` list.
- New per-step metrics: emit them from the relevant pool, surface in the `payload` dict in `LocalDriver.run`, and update `explorer/src/opd/types.ts` + the exporter together so the UI stays in sync.
