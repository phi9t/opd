from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

from opd.batches import TrainBatch
from opd.config import TrainConfig
from opd.loss.kl import reverse_kl_loss
from opd.models.synthetic_qwen3 import SyntheticQwen3Config, SyntheticQwen3ForCausalLM


class StudentActor:
    def __init__(self, config: TrainConfig, device: str = "cpu") -> None:
        self.device = torch.device(device)
        student_cfg = SyntheticQwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.student_hidden_size,
        )
        self.model = SyntheticQwen3ForCausalLM(student_cfg).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        traj = batch.trajectory
        teacher = batch.teacher

        prompt_ids = traj.prompt_ids.to(self.device)
        token_ids = traj.token_ids.to(self.device)
        full = torch.cat([prompt_ids, token_ids], dim=1)

        student_logprobs = self.model.logprobs_on_tokens(full, token_ids)
        teacher_logprobs = teacher.teacher_logprobs.to(self.device)

        prompt_len = prompt_ids.shape[1]
        mask = traj.attention_mask[:, prompt_len:].to(self.device).float()

        loss = reverse_kl_loss(student_logprobs, teacher_logprobs, mask)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=float("inf"))
        self.optimizer.step()

        with torch.no_grad():
            per_token_kl = student_logprobs - teacher_logprobs
            mean_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1)

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "mean_kl": float(mean_kl.item()),
        }
