import torch


def reverse_kl_loss(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Per-token reverse KL on sampled tokens, masked-mean over response positions.

    Computes :math:`\mathcal{L}_{\mathrm{KL},t} = \log \pi_S(y_t|s_t) - \log \pi_T(y_t|s_t)`
    and averages over the response mask. See `docs/tutorials/02_on_policy_distillation.md`
    for the derivation and the mode-seeking argument for using reverse rather than forward KL.
    """
    per_token = student_logprobs - teacher_logprobs
    if mask is not None:
        per_token = per_token * mask
        return per_token.sum() / mask.sum().clamp(min=1)
    return per_token.mean()
