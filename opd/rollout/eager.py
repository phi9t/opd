from __future__ import annotations

import torch
import torch.nn.functional as F

from opd.batches import TrajectoryBatch
from opd.models.synthetic_qwen3 import SyntheticQwen3Config, SyntheticQwen3ForCausalLM


class StudentRollout:
    def __init__(
        self,
        config: SyntheticQwen3Config,
        device: str = "cpu",
        max_new_tokens: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.model = SyntheticQwen3ForCausalLM(config).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> TrajectoryBatch:
        prompt_ids = prompt_ids.to(self.device)
        batch_size, _ = prompt_ids.shape

        generated: list[torch.Tensor] = []
        logprobs: list[torch.Tensor] = []
        current = prompt_ids

        for _ in range(self.max_new_tokens):
            logits = self.model(current).logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            next_token = logits.argmax(dim=-1, keepdim=True)
            next_logprob = log_probs.gather(1, next_token)
            generated.append(next_token)
            logprobs.append(next_logprob)
            current = torch.cat([current, next_token], dim=1)

        token_ids = torch.cat(generated, dim=1)
        student_logprobs = torch.cat(logprobs, dim=1)
        full = torch.cat([prompt_ids, token_ids], dim=1)
        attention_mask = torch.ones(batch_size, full.shape[1], dtype=torch.long, device=self.device)

        return TrajectoryBatch(
            prompt_ids=prompt_ids,
            token_ids=token_ids,
            student_logprobs=student_logprobs,
            attention_mask=attention_mask,
        )
