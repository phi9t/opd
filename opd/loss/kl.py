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
