from __future__ import annotations

import json
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_PATH = _REPO_ROOT / "explorer" / "public" / "data" / "index.json"

_GLOSSARY = {
    "opd": (
        "On-policy distillation. The student samples its own trajectories, a frozen "
        "teacher scores those same tokens, and the student is updated to minimize "
        "$\\mathrm{KL}(\\pi_S \\,\\|\\, \\pi_T)$ on the state distribution it actually visits. "
        "Combines RL's correct credit assignment with SFT's dense per-token signal. "
        "See `docs/tutorials/02_on_policy_distillation.md`."
    ),
    "reverse_kl": (
        "The per-token loss: $\\mathcal{L}_{\\mathrm{KL},t} = \\log \\pi_S(y_t \\mid s_t) - \\log \\pi_T(y_t \\mid s_t)$, "
        "averaged over response positions. Reverse (not forward) KL because the expectation is "
        "under the student — the student learns to put mass where the teacher does on its own "
        "state distribution rather than covering the teacher's entire support. Mode-seeking."
    ),
    "on_policy": (
        "Sampling the training data from the current student instead of from a static "
        "teacher-generated corpus. Avoids exposure bias: the student is supervised on the "
        "states it will actually reach at inference, not on states only the teacher would have "
        "visited. The fix is the same DAGGER-style correction used in imitation learning."
    ),
    "dense_reward": (
        "RL teaches $O(1)$ bits per episode — one terminal scalar reward spread across all "
        "tokens by the policy gradient. Distillation against a teacher's logprobs teaches "
        "$O(N)$ bits per episode — one signal per token. For a fixed compute budget, the "
        "per-token signal-to-noise ratio is dramatically higher."
    ),
    "weight_sync": (
        "Between actor and rollout each step: the student's updated parameters are copied into "
        "the rollout module so step $t{+}1$ samples from the post-update policy. Recorded as "
        "`sync_ms` and `sync_bytes`. Tiny tier: in-process `load_state_dict`. At scale: "
        "FSDP `state_dict` gather followed by a vLLM `load_weights` bulk call."
    ),
    "ppo": (
        "Proximal Policy Optimization. Clipped importance-sampling policy gradient with a value "
        "baseline. Objective: $\\mathbb{E}_t[\\min(r_t(\\theta) A_t,\\ \\mathrm{clip}(r_t,\\,1{-}\\epsilon,\\,1{+}\\epsilon) A_t)]$ "
        "where $r_t = \\pi_\\theta / \\pi_{\\theta_{\\mathrm{old}}}$. OPD with `epochs_per_step = 1` and "
        "$A_t = -\\mathcal{L}_{\\mathrm{KL},t}$ degenerates into a single-epoch PPO."
    ),
    "grpo": (
        "Group Relative Policy Optimization. Drops the value critic; advantages are normalized "
        "across $n$ rollouts per prompt: $A_t = (R - \\mathrm{mean}(R_{\\mathrm{group}})) / \\mathrm{std}(R_{\\mathrm{group}})$. "
        "Cheaper than PPO (no critic) at the cost of needing multiple rollouts per prompt."
    ),
    "token_kl_heatmap": (
        "Per-token reverse-KL rendered as a sentence with each token shaded by "
        "$|\\mathcal{L}_{\\mathrm{KL},t}| / P_{95}(|\\mathcal{L}_{\\mathrm{KL}}|)$. Magnitude (not sign) so debug spikes "
        "in either direction surface. The `_build_token_samples` selector picks the top-$N$ "
        "sequences by maximum per-token $|\\mathrm{KL}|$ each step."
    ),
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
