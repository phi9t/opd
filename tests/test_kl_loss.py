import torch
from opd.loss.kl import reverse_kl_loss


def test_reverse_kl_known_values():
    student = torch.tensor([-1.0, -2.0])
    teacher = torch.tensor([-1.5, -1.0])
    loss = reverse_kl_loss(student, teacher)
    expected = torch.tensor([-1.0 - (-1.5), -2.0 - (-1.0)]).mean()
    assert torch.allclose(loss, expected)
