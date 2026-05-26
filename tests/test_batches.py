import torch

from opd.batches import TrajectoryBatch, TeacherBatch, TrainBatch


def test_trajectory_roundtrip_dict():
    traj = TrajectoryBatch(
        prompt_ids=torch.tensor([[1, 2]]),
        token_ids=torch.tensor([[3, 4]]),
        student_logprobs=torch.tensor([[-1.0, -1.1]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        meta={"step": 0},
    )
    d = traj.to_dict()
    traj2 = TrajectoryBatch.from_dict(d)
    assert torch.equal(traj2.token_ids, traj.token_ids)
