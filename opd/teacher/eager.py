from __future__ import annotations

import torch

from opd.batches import TeacherBatch, TrajectoryBatch
from opd.config import TrainConfig
from opd.models.synthetic_qwen3 import SyntheticQwen3Config, SyntheticQwen3ForCausalLM


class TeacherScorer:
    def __init__(self, config: TrainConfig, device: str = "cpu") -> None:
        self.device = torch.device(device)
        teacher_cfg = SyntheticQwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.teacher_hidden_size,
        )
        self.model = SyntheticQwen3ForCausalLM(teacher_cfg).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def score(self, traj: TrajectoryBatch) -> TeacherBatch:
        prompt_ids = traj.prompt_ids.to(self.device)
        token_ids = traj.token_ids.to(self.device)
        full = torch.cat([prompt_ids, token_ids], dim=1)
        teacher_logprobs = self.model.logprobs_on_tokens(full, token_ids)
        return TeacherBatch(teacher_logprobs=teacher_logprobs)
