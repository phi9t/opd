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
