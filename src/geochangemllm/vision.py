from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import ChangeDecoder, SiameseEncoder, TemporalFusion


class VisionChangeModel(nn.Module):
    """Vision-only network used for unsupervised pretraining."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = SiameseEncoder(in_channels, base_channels)
        self.fusion = TemporalFusion(self.encoder.channels)
        self.decoder = ChangeDecoder(self.encoder.channels)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, Any]:
        before = self.encoder(t1)
        after = self.encoder(t2)
        fused = self.fusion(before, after)
        logits = self.decoder(fused, output_size=t1.shape[-2:])
        return {
            "mask_logits": logits,
            "before_deep": before[-1],
            "after_deep": after[-1],
        }


def load_vision_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> int:
    """Load encoder/fusion/decoder weights from unsupervised or full checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("vision", checkpoint.get("model", checkpoint))
    state = {
        (key.removeprefix("module.")): value
        for key, value in state.items()
    }
    loaded = 0
    for module_name in ("encoder", "fusion", "decoder"):
        prefix = f"{module_name}."
        module_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not module_state:
            raise ValueError(f"Checkpoint has no {module_name} weights: {checkpoint_path}")
        getattr(model, module_name).load_state_dict(module_state, strict=True)
        loaded += len(module_state)
    return loaded
