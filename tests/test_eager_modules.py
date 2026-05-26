import math

import torch

from opd.actor import StudentActor
from opd.batches import TrainBatch
from opd.config import TrainConfig
from opd.models.synthetic_qwen3 import SyntheticQwen3Config
from opd.rollout import StudentRollout
from opd.teacher import TeacherScorer


def _tiny_train_config() -> TrainConfig:
    return TrainConfig(
        tier="tiny",
        runtime="local",
        device="cpu",
        engine="eager",
        loss_mode="kl",
        teacher_signal="sampled",
        max_steps=1,
        batch_size=1,
        max_new_tokens=4,
        lr=1e-3,
        seed=0,
        run_dir="runs",
        student_hidden_size=32,
        teacher_hidden_size=64,
        vocab_size=64,
        prompts_path="data/prompts/tiny_stories.jsonl",
        tokenizer_dir="data/tokenizer/tiny",
    )


def test_eager_generate_score_train_step():
    cfg = _tiny_train_config()
    student_cfg = SyntheticQwen3Config(vocab_size=64, hidden_size=32, num_hidden_layers=2)

    rollout = StudentRollout(student_cfg, device="cpu", max_new_tokens=cfg.max_new_tokens)
    teacher = TeacherScorer(cfg, device="cpu")
    actor = StudentActor(cfg, device="cpu")

    prompt_ids = torch.tensor([[1, 2, 3]])
    traj = rollout.generate(prompt_ids)
    teach = teacher.score(traj)
    metrics = actor.train_step(TrainBatch(trajectory=traj, teacher=teach))

    assert traj.token_ids.shape == (1, cfg.max_new_tokens)
    assert traj.student_logprobs.shape == traj.token_ids.shape
    assert teach.teacher_logprobs.shape == traj.token_ids.shape
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["grad_norm"])
    assert math.isfinite(metrics["mean_kl"])
