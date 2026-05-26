# OPD Compute Infrastructure Design

## Purpose

Build a minimal, hackable **on-policy distillation (OPD)** training system in the `opd` repository for **research and teaching**. The system implements the production-shaped MLSys loop (student rollout → teacher logprob service → KL / OPD-RL update → weight sync) without wrapping verl or NeMo-RL, while making mechanisms legible on a single node before scaling to multi-node with the same interfaces.

Primary mental model (teaching):

```text
OPD = RL rollout stack + teacher-logprob service + KL / policy update
```

Compared to PPO RLHF (actor + rollout + reference + reward + critic + GAE + dual optimizers), pure OPD removes the critic, reward model, and sparse scalar reward path, and adds a dense teacher scoring service.

## Goals

- Run a **`tiny` mechanism demo on CPU** (no GPU, no Ray): synthetic Qwen3-style micro-models + bundled tiny-stories, full OPD loop in one process.
- Run **`qwen3_small` on 1×8 GPUs (~48GB)** with vLLM + FSDP + Ray three-pool layout (production-shaped path).
- Teach OPD dataflow first on CPU; then GPU placement, teacher bandwidth, and weight sync on the HF tier.
- Export run artifacts to a **clinique/explorer-style** static UI (Python JSON export → Vite/React, no backend).

## Non-Goals (v1)

- verl / NeMo-RL integration as the training spine.
- Async one-step / two-step schedulers (overlap rollout, teacher, train).
- Hybrid OPD + task rewards (PPO/GRPO loss terms).
- PPO/GRPO baseline trainers in-repo.
- Full-vocabulary teacher logits.
- Multi-node execution (interfaces only; implementation in M4).
- Requiring GPU or Ray for the `tiny` tier (CPU + local driver is the default).

## Requirements Summary

| Area | Decision |
|------|----------|
| Audience | Research + teaching |
| Scale | Single-node v1; multi-node v2, same APIs |
| Stack | `tiny`: eager PyTorch on CPU, single-process train; `qwen3_small`: vLLM + FSDP |
| Orchestration | `tiny`: **local sync driver** (default); `qwen3_small`: Ray placement groups, three pools |
| Scheduler | Strictly synchronous steps |
| Teacher signal | Sampled-token default; top-k behind flag |
| Loss | `kl` default; Thinking Machines-style `opd_rl` behind flag |
| Data | `tiny`: bundled tiny-stories JSONL; `qwen3_small`: HF loader + recipe |
| Observability | In-repo explorer UI + structured step metrics |
| Lab hardware | `tiny`: any CPU machine; `qwen3_small`: 1×8 ~48GB |

## Architecture

### Runtime modes

Two layouts share the **same step semantics** (`gen → teacher → train → sync`) and batch types; only orchestration differs.

**`tiny` default — local CPU driver (`runtime=local`):**

```text
┌──────────────────────────────────────────────────────────┐
│  LocalDriver (single process, device=cpu)                │
│  student_rollout → teacher → student_actor → weight_sync │
│  (three modules, one Python process, sequential calls)   │
└──────────────────────────────────────────────────────────┘
```

**`qwen3_small` — Ray three-pool GPU layout (`runtime=ray`):**

```text
┌─────────────────────────────────────────────────────────┐
│  RayDriver (sync scheduler)                             │
│  for step: gen → teacher → train → sync                 │
└────────────┬────────────────┬────────────────┬──────────┘
    ┌────────▼────────┐ ┌─────▼──────┐ ┌──────▼─────────┐
    │ ActorPool       │ │ RolloutPool│ │ TeacherPool    │
    │ FSDP student    │ │ vLLM       │ │ vLLM (frozen)  │
    └────────┬────────┘ └─────┬──────┘ └──────┬─────────┘
             └──── weight_sync (student → rollout) ───────┘
```

Ray **placement groups** apply only when `runtime=ray`. Optional `tiny` + `runtime=ray` + `device=cuda` reuses the three-pool diagram for placement-group labs (not required for M1).

### Config tiers

| Tier | Purpose | Models | Runtime | Device | Defaults |
|------|---------|--------|---------|--------|----------|
| `tiny` | Mechanism demo | In-repo **synthetic Qwen3-style** 2-layer transformers; see [Synthetic tiny tier](#synthetic-tiny-tier) | `local` | **`cpu`** | `engine=eager`, `loss=kl`, `teacher.signal=sampled` |
| `qwen3_small` | Real-model experiments | HF: `Qwen/Qwen3-0.6B` (student), `Qwen/Qwen3-1.7B` (teacher) | `ray` | `cuda` | `engine=vllm`, `loss=kl`; `topk` / `opd_rl` via flags |

Tier YAML files live under `configs/`. All tiers share the same **step contract**, batch types, and loss modules; `runtime`, `device`, and engine backends differ.

### Synthetic tiny tier

The `tiny` tier exists to exercise the full OPD control plane without Hugging Face downloads, vLLM custom-model registration, or long generation. Models are **defined in-repo**, not pulled from the Hub.

**Architecture (`opd/models/synthetic_qwen3.py`):**

- Qwen3-**style** decoder stack (RMSNorm, SwiGLU MLP, RoPE, GQA attention) with configurable micro-dims.
- Default **student:** `num_hidden_layers=2`, `hidden_size=256`, `num_attention_heads=4`, `num_key_value_heads=2`, `vocab_size=8192`.
- Default **teacher:** same layer count, **`hidden_size=512`** (wider MLP/attention), frozen during OPD.
- Shared tokenizer: in-repo **byte-level or BPE vocab** built from the story corpus (fixed `vocab.json` + `merges` or simplified unigram table shipped under `data/tokenizer/tiny/`).

**Initialization:**

- Fixed global seed; student and teacher start from independent random init (teacher may optionally load a one-time **fixture SFT checkpoint** produced by `scripts/build_tiny_teacher_fixture.py` so teacher logprobs are non-degenerate — still fully local, no Hub).

**Data (`data/prompts/tiny_stories.jsonl`):**

- Bundled **synthetic tiny-story** prompts: short narrative prefixes (~32–128 tokens target continuation).
- Generated offline by a checked-in script (`scripts/gen_tiny_stories.py`) for reproducibility; committed JSONL in repo.
- Training reads prompt column only; completions come from on-policy rollout.

**CPU-only default (`device=cpu`, `runtime=local`):**

- Single Python process holds three logical modules: `StudentRollout`, `TeacherScorer`, `StudentActor` (train).
- All tensors on CPU; batch sizes small (e.g. 4 prompts, 1 rollout each, ≤64 response tokens).
- No Ray, no FSDP, no vLLM, no CUDA required. Runnable on a laptop for lectures and CI.
- Same `steps/<n>.json` metrics (`gen_ms`, `teacher_ms`, `train_ms`, `sync_ms`) so the explorer UI works identically.

**Why not vLLM for `tiny`:** vLLM does not practically serve arbitrary 2-layer custom architectures. `tiny` uses **eager** `forward` + manual generate for rollout and teacher scoring.

**Weight sync (`tiny`, local):** after `optimizer.step()`, `student_rollout.load_state_dict(student_actor.state_dict())` in-process; still record `sync_ms` / `sync_bytes` for teaching.

**Optional GPU path (`device=cuda`, `runtime=ray`):** same synthetic models with Ray three-pool + eager engines; for placement-group demos only (post-M1 or flagged experimental).

**Runtime budget:** target **&lt;2 minutes** for default 20-step CPU demo; **&lt;5 minutes** for 50-step CPU run.

### Multi-node (v2 / M4)

- Same Ray actor classes and driver API.
- Placement groups span nodes; `cluster.gpus_per_pool` and network collectives scale.
- No change to `TrajectoryBatch` / `TeacherBatch` / exporter schema.

## Repository Layout

```text
opd/
  runtime/          # local_driver.py, ray_driver.py, placement
  models/           # synthetic_qwen3.py (+ config dataclasses)
  rollout/          # vllm.py, eager.py → TrajectoryBatch
  teacher/          # vllm.py, eager.py → TeacherBatch
  actor/            # FSDP train step, weight_sync (vllm + eager targets)
  loss/             # kl.py, opd_rl.py
  data/             # loaders, tokenizer helpers
  export/           # explorer bundle builder
configs/
  tier_tiny.yaml    # runtime=local, device=cpu, engine=eager, tiny_stories
  tier_qwen3_small.yaml
explorer/           # Vite/React static dashboard
scripts/
  run_lab.sh
  gen_tiny_stories.py
  build_tiny_teacher_fixture.py   # optional local teacher SFT
docs/tutorials/
  01_three_pools.md
  02_sync_step_trace.md
  03_kl_vs_opd_rl.md
data/
  prompts/tiny_stories.jsonl
  tokenizer/tiny/                 # vocab for synthetic tier
fixtures/
  tiny_teacher_sft/               # optional prebuilt teacher ckpt
```

## Core Components

| Module | Responsibility |
|--------|----------------|
| `runtime.local_driver` | CPU single-process sync loop (`tiny` default) |
| `runtime.ray_driver` | Ray sync loop (`qwen3_small`) |
| `runtime.placement` | GPU bundles when `runtime=ray` |
| `models.synthetic_qwen3` | Tiny Qwen3-style student/teacher factories |
| `rollout.worker` | Dispatch `engine` → `eager` or `vllm` → `TrajectoryBatch` |
| `teacher.worker` | Dispatch `engine` → `eager` or `vllm` → `TeacherBatch` |
| `actor.worker` | FSDP forward/backward, optimizer |
| `actor.weight_sync` | FSDP state → rollout engine (`load_state_dict` or vLLM bulk load) |
| `loss.kl` | Token KL (sampled or top-k sparse) |
| `loss.opd_rl` | Clipped IS update on dense KL advantage |
| `data.*` | Prompt loading, HF datasets, recipes |
| `export.explorer` | Static JSON bundles for UI |

### Shared batch types

All loss modes consume the same scored batch. Implementations must not fork the Ray pipeline per loss.

```python
@dataclass
class TrajectoryBatch:
    prompt_ids: ...
    token_ids: ...          # generated continuation
    student_logprobs: ...   # at sampled positions
    attention_mask: ...
    meta: dict              # step, run_id, lengths

@dataclass
class TeacherBatch:
    # sampled mode
    teacher_logprobs: ...   # log pi_T(y_t | s_t) for student y_t
    # topk mode (optional)
    topk_ids: ...
    topk_logprobs: ...

@dataclass
class TrainBatch:
    trajectory: TrajectoryBatch
    teacher: TeacherBatch
    old_logprobs: ...       # required when loss.mode == opd_rl
```

### Configuration flags

| Flag | Values | Default |
|------|--------|---------|
| `runtime` | `local`, `ray` | `local` for `tiny`, `ray` for `qwen3_small` |
| `device` | `cpu`, `cuda` | `cpu` for `tiny`, `cuda` for `qwen3_small` |
| `engine` | `eager`, `vllm` | `eager` for `tiny`, `vllm` for `qwen3_small` |
| `loss.mode` | `kl`, `opd_rl` | `kl` |
| `teacher.signal` | `sampled`, `topk` | `sampled` (`topk` only when `engine=vllm` in v1) |
| `teacher.topk` | int | 128 (when `topk`) |
| `loss.kl_direction` | `reverse`, `forward` | `reverse` |
| `opd_rl.clip_eps` | float | 0.2 |
| `opd_rl.epochs_per_step` | int | 1 |

## Synchronous Training Step

```text
1. rollout.generate(prompt_batch)  → TrajectoryBatch
2. teacher.score(trajectory)      → TeacherBatch
3. actor.train(batch, loss_mode)  → loss, grad_norm, aux metrics
4. actor.weight_sync(rollout_ref) → push student weights to vLLM
5. driver.log_step(); optional checkpoint
```

**On-policy semantics (v1):** rollouts at step `t` use student weights from the end of step `t-1`. No pipelining or stale-policy overlap.

**Per-step artifacts:**

```text
runs/<run_id>/
  config.yaml
  metrics.jsonl
  steps/<step>.json
  checkpoints/...
```

`steps/<n>.json` fields (minimum):

- Phase timings: `gen_ms`, `teacher_ms`, `train_ms`, `sync_ms`
- `sync_bytes`
- Token counts, mean response length
- `mean_kl`, `loss`, `grad_norm`
- `loss_mode`, `teacher_signal`
- If `opd_rl`: `clip_fraction`, `mean_ratio`

## Loss Modules

### `kl` (default)

Per-token reverse KL on sampled tokens:

\[
\mathcal{L}_{\text{KL},t} = \log \pi_S(y_t \mid s_t) - \log \pi_T(y_t \mid s_t)
\]

Top-k mode: sparse KL over teacher support set (gather student logits for `topk_ids`). v1 may use full-logit gather on actor for small tiers; document TP-aware gather as v2 improvement.

### `opd_rl` (flag)

Dense advantage from teacher signal:

\[
A_t = -(\log \pi_S(y_t \mid s_t) - \log \pi_T(y_t \mid s_t))
\]

PPO-style clipped importance sampling on student rollouts using `old_logprobs` captured at generation time. Configurable `epochs_per_step` (default 1).

## Weight Sync

First-class phase between actor and rollout pools.

- **`engine=eager`:** gather FSDP `state_dict` → `rollout.load_state_dict` (same architecture module).
- **`engine=vllm`:** gather FSDP state → vLLM `load_weights` bulk API.
- Expose `sync_ms` and `sync_bytes` in step metrics (teaching: compare to HybridFlow ~140GB/iter resharding narrative at scale on `qwen3_small`).

Document sharding layout mismatch between FSDP and vLLM TP as an intentional lecture topic on the HF tier.

## Explorer (Observability UI)

Follow **clinique/explorer** pattern:

- Python is source of truth for metrics aggregation and bundle shape.
- React displays static JSON only; no scorer reimplementation in TS.
- GitHub Pages compatible; committed demo bundle for CI/Playwright.

### Export path

```bash
# after training
opd export-explorer --run <run_id> --out explorer/public/data/runs/<run_id>.json

cd explorer && npm install && npm run dev
```

### Bundle shape (v1)

One self-contained JSON file per run:

```text
explorer/public/data/runs/<run_id>.json
explorer/public/data/index.json          # lists available runs for the UI
```

`runs/<run_id>.json` includes: run metadata, tier, HF model ids, loss flags, `steps[]` with phase timings, KL/loss series, glossary snippets (OPD vs PPO vs GRPO component table).

### UI panels (v1)

1. **Run overview** — tier, models, wall time, steps, final KL.
2. **Step timeline** — stacked gen / teacher / train / sync.
3. **Learning curves** — KL, loss, grad norm, response length.
4. **OPD-RL panel** — clip fraction, IS ratio histogram (when enabled).
5. **Glossary** — MLSys comparison table for teaching.

Commit one completed `tiny` tier demo JSON under `explorer/public/data/` for offline UI development.

## Data

| Source | Tier | Use |
|--------|------|-----|
| `data/prompts/tiny_stories.jsonl` | `tiny` | Bundled synthetic story prompts (<1k lines) |
| `data/tokenizer/tiny/` | `tiny` | Shared vocab for student + teacher |
| `data/loaders/hf.py` | `qwen3_small` | HF dataset name, split, column mapping |
| `configs/tier_qwen3_small.yaml` | `qwen3_small` | HF model ids, prompt recipe, hyperparameters |

**Tokenizer constraint:** student and teacher must share the same vocabulary table per tier. On `tiny`, both use `data/tokenizer/tiny/`. On `qwen3_small`, both must be the same Qwen3 model family. Fail fast at init if `vocab_size` differs.

## Failure Handling

| Condition | Behavior |
|-----------|----------|
| Tokenizer / vocab mismatch | `RuntimeError` at init |
| Teacher OOM | Log batch size; suggest reduce `batch_size` or `max_tokens` |
| Rollout engine stale weights | Assert weight version / step counter after sync |
| Ray actor death | Surface actor id + stderr; no silent retry in v1 (`runtime=ray` only) |
| CUDA unavailable when `device=cuda` | Fail fast with clear message |
| Invalid flag combo | Validate config before cluster allocation |

## Testing

| Layer | Scope |
|-------|--------|
| Unit | `loss.kl`, `loss.opd_rl` on synthetic tensors; batch serde |
| Integration | `tiny` CPU: 2-step run in CI (no GPU) |
| E2E | `qwen3_small`: 1 step on 8 GPU (nightly or manual) |
| Explorer | Playwright against committed demo JSON |

## Milestones

| ID | Deliverable |
|----|-------------|
| M1 | `tiny` CPU local driver + synthetic models + `tiny_stories` E2E + `export-explorer` + tutorial 01 + demo JSON |
| M2 | `qwen3_small` recipe + `teacher.signal=topk` |
| M3 | `loss.mode=opd_rl` + tutorial 03 |
| M4 | Multi-node placement groups, same driver API |

## Compute Comparison (Teaching Reference)

Per-step cost sketch (B prompts, n rollouts/prompt, T tokens, α ≈ train/inference forward ratio):

| Method | Dominant extra infra vs SFT |
|--------|----------------------------|
| SFT | None |
| OPD | `C_s BnT` gen + `C_t BnT` teacher + `α C_s BnT` train + sync |
| PPO | Above + reward + ref + critic paths, GAE, critic train |
| GRPO | Gen + verifier + ref; large n; no critic |

OPD v1 intentionally surfaces **teacher wait** and **sync** in the explorer timeline so students can relate to `wait_prev_teacher` operations guidance in production async OPD systems (verl), even though v1 is synchronous.

## References

- Thinking Machines — On-Policy Distillation (sampled-token, dense per-token signal)
- verl — async OPD, top-k KL, schedulers (reference only, not a dependency)
- HybridFlow — RLHF multi-model layout and weight resharding
- clinique/explorer — static export + React explorer pattern

## Approval

Design approved in brainstorming session 2026-05-26. Implementation proceeds via `writing-plans` after spec review.
