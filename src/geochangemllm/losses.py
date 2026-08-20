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


def temporal_difference_score(
    t1: torch.Tensor,
    t2: torch.Tensor,
    smoothing_kernel: int = 5,
) -> torch.Tensor:
    """Return a smoothed, single-channel radiometric change score."""
    score = (t1 - t2).abs().mean(dim=1, keepdim=True)
    if smoothing_kernel > 1:
        padding = smoothing_kernel // 2
        score = F.avg_pool2d(score, smoothing_kernel, stride=1, padding=padding)
    return score


def pseudo_change_targets(
    t1: torch.Tensor,
    t2: torch.Tensor,
    change_quantile: float = 0.85,
    stable_quantile: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build high-confidence pseudo labels from an aligned image pair.

    Pixels above the per-image change quantile are positive, pixels below the
    stable quantile are negative, and the uncertain middle interval is ignored.
    Constant pairs are treated as entirely stable.
    """
    score = temporal_difference_score(t1, t2)
    targets, confidence = pseudo_targets_from_score(
        score,
        change_quantile=change_quantile,
        stable_quantile=stable_quantile,
    )
    return targets, confidence, score


def pseudo_targets_from_score(
    score: torch.Tensor,
    change_quantile: float = 0.85,
    stable_quantile: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < stable_quantile < change_quantile < 1.0:
        raise ValueError("Require 0 < stable_quantile < change_quantile < 1")
    flat = score.flatten(1)
    high = torch.quantile(flat, change_quantile, dim=1).view(-1, 1, 1, 1)
    low = torch.quantile(flat, stable_quantile, dim=1).view(-1, 1, 1, 1)
    constant = high <= 1e-6
    targets = (score >= high).to(score.dtype)
    confidence = ((score >= high) | (score <= low)).to(score.dtype)
    targets = torch.where(constant, torch.zeros_like(targets), targets)
    confidence = torch.where(constant, torch.ones_like(confidence), confidence)
    return targets, confidence


def sequence_pseudo_change_targets(
    frames: torch.Tensor,
    frame_mask: torch.Tensor,
    change_quantile: float = 0.85,
    stable_quantile: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pseudo labels from the maximum valid adjacent-frame change."""
    pair_scores = torch.stack(
        [
            temporal_difference_score(frames[:, index], frames[:, index + 1])
            for index in range(frames.shape[1] - 1)
        ],
        dim=1,
    )
    pair_mask = frame_mask[:, :-1] & frame_mask[:, 1:]
    masked_scores = pair_scores.masked_fill(~pair_mask[:, :, None, None, None], 0.0)
    score = masked_scores.max(dim=1).values
    targets, confidence = pseudo_targets_from_score(
        score,
        change_quantile=change_quantile,
        stable_quantile=stable_quantile,
    )
    return targets, confidence, score


def pseudo_change_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    confidence: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Confidence-masked BCE and Dice loss for weak/unsupervised masks."""
    pixel_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    confident_pixels = confidence.sum().clamp_min(1.0)
    bce = (pixel_bce * confidence).sum() / confident_pixels
    probabilities = logits.sigmoid() * confidence
    masked_targets = targets * confidence
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * masked_targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + masked_targets.sum(dim=dimensions)
    dice = 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
    total = bce + dice
    return total, {"pseudo_bce": bce.detach(), "pseudo_dice": dice.detach()}


def stable_feature_consistency(
    before: torch.Tensor,
    after: torch.Tensor,
    score: torch.Tensor,
    stable_quantile: float = 0.50,
) -> torch.Tensor:
    """Align normalized deep features only in radiometrically stable regions."""
    threshold = torch.quantile(score.flatten(1), stable_quantile, dim=1).view(-1, 1, 1, 1)
    stable = (score <= threshold).to(before.dtype)
    stable = F.interpolate(stable, size=before.shape[-2:], mode="nearest")
    before = F.normalize(before, dim=1)
    after = F.normalize(after, dim=1)
    difference = (before - after).square().mean(dim=1, keepdim=True)
    return (difference * stable).sum() / stable.sum().clamp_min(1.0)


def temporal_symmetry_loss(forward_logits: torch.Tensor, reverse_logits: torch.Tensor) -> torch.Tensor:
    """Binary change probability should be invariant to time-pair order."""
    return F.mse_loss(forward_logits.sigmoid(), reverse_logits.sigmoid())
