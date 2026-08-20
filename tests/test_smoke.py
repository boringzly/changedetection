from __future__ import annotations

import torch

from geochangemllm.losses import joint_loss
from geochangemllm.model import GeoChangeMLLM


def test_joint_forward_backward() -> None:
    model = GeoChangeMLLM(base_channels=8, text_model="tiny", token_grid=2)
    t1 = torch.rand(2, 3, 64, 64)
    t2 = torch.rand(2, 3, 64, 64)
    masks = (torch.rand(2, 1, 64, 64) > 0.8).float()
    output = model(
        t1,
        t2,
        torch.tensor([0.5, 10.0]),
        ["forest", "disaster"],
        ["林地减少", "新增滑坡区域"],
    )
    loss, _ = joint_loss(output["mask_logits"], masks, output["text_loss"], 0.2)
    loss.backward()
    assert output["mask_logits"].shape == masks.shape
    assert model.token_projector.project[1].weight.grad is not None
