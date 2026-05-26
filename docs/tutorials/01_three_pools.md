# Tutorial 01: Three Pools on CPU

This walkthrough runs the full **on-policy distillation (OPD)** loop on the `tiny` tier: synthetic Qwen3-style models, bundled tiny-stories prompts, and a single-process **local driver**. The goal is to see the same four phases you will later split across Ray placement groups on GPU, without needing CUDA or Ray.

## Mental model: OPD = rollout + teacher + train + sync

Each training step is strictly synchronous:

```text
gen → teacher → train → sync
```

| Phase | Module (tiny tier) | What happens |
|-------|-------------------|--------------|
| **Rollout** (`gen_ms`) | `StudentRollout` | On-policy generation from the student; produces token trajectories and logprobs. |
| **Teacher** (`teacher_ms`) | `TeacherScorer` | Frozen wider teacher scores the same tokens (sampled-token logprobs by default). |
| **Train** (`train_ms`) | `StudentActor` | KL (or `opd_rl`) loss + optimizer step on the student. |
| **Sync** (`sync_ms`) | weight copy | Student weights are copied into the rollout module so step `t+1` uses the updated policy. |

Compared to PPO RLHF, pure OPD drops the critic, reward model, and sparse scalar reward path, and replaces them with a **dense teacher logprob service**.

## Three logical pools, one process

On `tiny` with `runtime=local` and `device=cpu`, there is no Ray and no separate machines. `LocalDriver` holds three modules in one Python process and calls them sequentially:

```text
┌──────────────────────────────────────────────────────────┐
│  LocalDriver (single process, device=cpu)                  │
│  StudentRollout → TeacherScorer → StudentActor → sync    │
└──────────────────────────────────────────────────────────┘
```

This mirrors the **qwen3_small** layout where Ray places **RolloutPool**, **TeacherPool**, and **ActorPool** on different GPU groups—but here everything is colocated for teaching and CI.

Default config: `configs/tier_tiny.yaml` (`max_steps: 20`, `batch_size: 4`, `max_new_tokens: 32`).

## Run the lab

From the repo root:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
./scripts/run_lab.sh
```

`run_lab.sh` runs:

```bash
.venv/bin/opd train --config configs/tier_tiny.yaml
```

Artifacts land under `runs/run_<timestamp>/`:

- `config.yaml` — frozen training config
- `steps/<n>.json` — per-step metrics (phase timings, KL, loss, etc.)
- `metrics.jsonl` — append-only log of the same payloads

Pick the latest run directory (e.g. `ls -t runs | head -1`) for export.

### Export to the explorer

```bash
.venv/bin/opd export-explorer \
  --run runs/<latest_run_dir> \
  --out explorer/public/data/runs/my-run.json
```

This builds a self-contained JSON bundle and updates `explorer/public/data/index.json`.

### Open the UI

```bash
cd explorer && npm install && npm run dev
```

Load your run from the index. Panels include:

- **Run overview** — tier, models, wall time, aggregate phase cost
- **Step timeline** — stacked `gen_ms` / `teacher_ms` / `train_ms` / `sync_ms`
- **Learning curves** — KL, loss, grad norm, response length
- **Glossary** — OPD vs PPO vs GRPO (from the exported bundle)

A committed demo bundle is at `explorer/public/data/runs/tiny-demo.json` for offline UI work without training.

## Step JSON → MLSys bottlenecks

Each `steps/<n>.json` records wall-clock milliseconds per phase. Map them to production concerns:

| Field | MLSys bottleneck | What to look for |
|-------|------------------|------------------|
| `gen_ms` | **Rollout / inference throughput** | Dominant on-policy cost; scales with `batch_size`, `max_new_tokens`, and model size. Async OPD systems overlap this with other pools; v1 is sync so gen often tops the stack. |
| `teacher_ms` | **Teacher service bandwidth** | Extra forward passes on a larger frozen model; at scale this becomes `wait_prev_teacher`-style queueing. Wider teacher (`teacher_hidden_size` > student) shows up here. |
| `train_ms` | **Backward / optimizer** | FSDP or multi-GPU train; on `tiny` it is a single CPU backward. Roughly α× forward cost in large setups. |
| `sync_ms` | **Weight resharding** | Copying actor weights into the rollout engine. Tiny tier: in-process `load_state_dict`. HF tier: FSDP gather → vLLM bulk load; `sync_bytes` teaches HybridFlow-style resharding cost. |

Other useful fields:

- `sync_bytes` — bytes moved on sync (teaching resharding volume)
- `mean_kl`, `loss`, `grad_norm` — optimization health
- `num_tokens`, `mean_response_length` — throughput normalization

On CPU `tiny` runs, `gen_ms` is usually largest; `teacher_ms` is smaller but still visible because the teacher is wider. After scaling to `qwen3_small`, expect `teacher_ms` and `sync_ms` to matter more relative to gen.

## OPD vs PPO vs GRPO (teaching)

Per-step **infrastructure** sketch (B prompts, n rollouts/prompt, T tokens):

| Method | Dominant extra infra vs SFT |
|--------|----------------------------|
| **SFT** | None |
| **OPD** | Student gen + teacher forwards + student train + weight sync |
| **PPO** | OPD-like rollout stack **plus** reference model, reward model, **critic**, GAE, and critic training — often **five model roles** in HybridFlow-style RLHF |
| **GRPO** | Generation + verifier/reference paths; large **n** per prompt; **no critic** (group-relative advantages) |

**OPD (this repo, v1):** student rollout, frozen teacher, KL update, sync. No critic, no reward model, no sparse verifier scalar—dense per-token signal from teacher logprobs.

**PPO:** keeps the multi-model RLHF layout (actor, rollout, reference, reward, critic) and clipped policy-gradient updates with a value baseline.

**GRPO:** drops the critic and normalizes advantages within prompt groups; still needs generation and reference/verifier paths at scale.

The explorer timeline intentionally surfaces **teacher wait** and **sync** even though v1 is synchronous—so you can relate step timings to async OPD schedulers (e.g. verl) in production.

## Next steps

- **M2:** `qwen3_small` tier — Ray three-pool layout, vLLM rollout, FSDP actor, real HF models.
- **Tutorial 02:** sync step trace and `sync_bytes` at scale.
- **Tutorial 03:** `loss.mode=opd_rl` vs KL.

Specs and plans: [docs/superpowers/specs/2026-05-26-opd-compute-infra-design.md](../superpowers/specs/2026-05-26-opd-compute-infra-design.md).
