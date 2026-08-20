from __future__ import annotations

import torch
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = logits.sigmoid()
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    score = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - score.mean()


def joint_loss(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    text_loss: torch.Tensor,
    text_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bce = F.binary_cross_entropy_with_logits(mask_logits, masks)
    dice = dice_loss(mask_logits, masks)
    total = bce + dice + text_weight * text_loss
    return total, {"bce": bce.detach(), "dice": dice.detach(), "text": text_loss.detach()}
