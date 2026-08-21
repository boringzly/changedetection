from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .text import TextAdapter
from .vl import QwenVLAdapter


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class SiameseEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.channels = channels
        self.stages = nn.ModuleList(
            [
                ConvBlock(in_channels, channels[0], stride=2),
                ConvBlock(channels[0], channels[1], stride=2),
                ConvBlock(channels[1], channels[2], stride=2),
                ConvBlock(channels[2], channels[3], stride=2),
            ]
        )

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in self.stages:
            image = stage(image)
            features.append(image)
        return features


class TemporalFusion(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(channel * 4, channel, kernel_size=1) for channel in channels]
        )

    def forward(
        self, before: Sequence[torch.Tensor], after: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        fused = []
        for first, second, projection in zip(before, after, self.projections, strict=True):
            features = torch.cat((first, second, (first - second).abs(), first * second), dim=1)
            fused.append(projection(features))
        return fused


class ChangeDecoder(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.decode2 = ConvBlock(c3 + c2, c2)
        self.decode1 = ConvBlock(c2 + c1, c1)
        self.decode0 = ConvBlock(c1 + c0, c0)
        self.head = nn.Conv2d(c0, 1, kernel_size=1)

    def forward(self, features: Sequence[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        f0, f1, f2, f3 = features
        value = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode2(torch.cat((value, f2), dim=1))
        value = F.interpolate(value, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode1(torch.cat((value, f1), dim=1))
        value = F.interpolate(value, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode0(torch.cat((value, f0), dim=1))
        value = F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
        return self.head(value)


class ChangeTokenProjector(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, token_grid: int) -> None:
        super().__init__()
        self.token_grid = token_grid
        self.project = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gsd_project = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, feature: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature, (self.token_grid, self.token_grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.project(tokens)
        scale = torch.log1p(gsd.clamp_min(0.01)).view(-1, 1)
        scale_token = self.gsd_project(scale).unsqueeze(1)
        return self.norm(torch.cat((scale_token, tokens), dim=1))


class OrderedChangeTokenProjector(nn.Module):
    """Project every adjacent-pair feature while retaining temporal order."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        token_grid: int,
        max_pairs: int = 31,
    ) -> None:
        super().__init__()
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        self.token_grid = token_grid
        self.max_pairs = max_pairs
        self.project = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gsd_project = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.time_embedding = nn.Embedding(max_pairs, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        pair_features: torch.Tensor,
        pair_mask: torch.Tensor,
        gsd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_features.ndim != 5:
            raise ValueError("pair_features must have shape [batch,pairs,channels,height,width]")
        batch, pairs, channels, height, width = pair_features.shape
        del height, width
        if pair_mask.shape != (batch, pairs):
            raise ValueError("pair_mask must match pair_features [batch,pairs]")
        if pairs > self.max_pairs:
            raise ValueError(f"received {pairs} pairs, configured maximum is {self.max_pairs}")

        flattened = pair_features.reshape(batch * pairs, channels, *pair_features.shape[-2:])
        pooled = F.adaptive_avg_pool2d(flattened, (self.token_grid, self.token_grid))
        tokens_per_pair = self.token_grid**2
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.project(tokens).reshape(batch, pairs, tokens_per_pair, -1)
        positions = self.time_embedding(
            torch.arange(pairs, device=pair_features.device)
        ).view(1, pairs, 1, -1)
        tokens = (tokens + positions).reshape(batch, pairs * tokens_per_pair, -1)
        token_mask = pair_mask.repeat_interleave(tokens_per_pair, dim=1)

        scale = torch.log1p(gsd.clamp_min(0.01)).view(-1, 1)
        scale_token = self.gsd_project(scale).unsqueeze(1)
        tokens = self.norm(torch.cat((scale_token, tokens), dim=1))
        scale_mask = torch.ones((batch, 1), dtype=torch.bool, device=pair_mask.device)
        token_mask = torch.cat((scale_mask, token_mask), dim=1)
        return tokens, token_mask


class SequenceTemporalAggregator(nn.Module):
    """Aggregate adjacent-pair features across a variable-length sequence."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(channel * 2, channel, kernel_size=1) for channel in channels]
        )

    def forward(
        self,
        pair_features: Sequence[torch.Tensor],
        pair_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        aggregated = []
        mask = pair_mask[:, :, None, None, None]
        for features, projection in zip(pair_features, self.projections, strict=True):
            weights = mask.to(features.dtype)
            mean = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            minimum = torch.finfo(features.dtype).min
            maximum = features.masked_fill(~mask, minimum).max(dim=1).values
            aggregated.append(projection(torch.cat((mean, maximum), dim=1)))
        return aggregated


class GeoChangeMLLM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        text_model: str = "tiny",
        lora_rank: int = 0,
        freeze_text: bool = False,
        token_grid: int = 4,
        max_text_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.text = TextAdapter(text_model, max_text_tokens=max_text_tokens, lora_rank=lora_rank)
        if freeze_text:
            self.text.freeze_model()
        self.encoder = SiameseEncoder(in_channels, base_channels)
        self.fusion = TemporalFusion(self.encoder.channels)
        self.decoder = ChangeDecoder(self.encoder.channels)
        self.token_projector = ChangeTokenProjector(
            self.encoder.channels[-1], self.text.hidden_size, token_grid
        )

    @staticmethod
    def prompts(domains: Sequence[str], gsds: torch.Tensor) -> list[str]:
        return [
            f"你是遥感变化分析助手。业务={domain}，GSD={float(gsd):.2f}米。描述两期影像的主要变化："
            for domain, gsd in zip(domains, gsds.detach().cpu(), strict=True)
        ]

    def encode_change(
        self, t1: torch.Tensor, t2: torch.Tensor, gsd: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        before = self.encoder(t1)
        after = self.encoder(t2)
        fused = self.fusion(before, after)
        mask_logits = self.decoder(fused, output_size=t1.shape[-2:])
        tokens = self.token_projector(fused[-1], gsd)
        return mask_logits, tokens

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        gsd: torch.Tensor,
        domains: Sequence[str],
        descriptions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        mask_logits, tokens = self.encode_change(t1, t2, gsd)
        text_loss = self.text.language_loss(tokens, self.prompts(domains, gsd), descriptions)
        return {"mask_logits": mask_logits, "text_loss": text_loss, "visual_tokens": tokens}

    @torch.no_grad()
    def predict(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        gsd: torch.Tensor,
        domains: Sequence[str],
        generate_text: bool = False,
    ) -> tuple[torch.Tensor, list[str]]:
        mask_logits, tokens = self.encode_change(t1, t2, gsd)
        descriptions = []
        if generate_text:
            descriptions = self.text.generate(tokens, self.prompts(domains, gsd))
        return mask_logits.sigmoid(), descriptions


class GeoSequenceChangeMLLM(nn.Module):
    """All-frame weakly supervised model for TIF temporal sequences."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        text_model: str = "tiny",
        lora_rank: int = 0,
        freeze_text: bool = False,
        token_grid: int = 4,
        max_text_tokens: int = 128,
        vl_model: str | None = None,
        ordered_change_tokens: bool = False,
        max_change_pairs: int = 31,
        vl_min_pixels: int = 50_176,
        vl_max_pixels: int = 200_704,
    ) -> None:
        super().__init__()
        self.vl_model = vl_model
        self.ordered_change_tokens = ordered_change_tokens or vl_model is not None
        if vl_model is None:
            self.text = TextAdapter(
                text_model,
                max_text_tokens=max_text_tokens,
                lora_rank=lora_rank,
            )
        else:
            self.text = QwenVLAdapter(
                vl_model,
                max_text_tokens=max_text_tokens,
                lora_rank=lora_rank,
                freeze_vision=True,
                min_pixels=vl_min_pixels,
                max_pixels=vl_max_pixels,
            )
        if freeze_text:
            self.text.freeze_model()
        self.encoder = SiameseEncoder(in_channels, base_channels)
        self.fusion = TemporalFusion(self.encoder.channels)
        self.sequence_aggregator = SequenceTemporalAggregator(self.encoder.channels)
        self.decoder = ChangeDecoder(self.encoder.channels)
        if self.ordered_change_tokens:
            self.token_projector = OrderedChangeTokenProjector(
                self.encoder.channels[-1],
                self.text.hidden_size,
                token_grid,
                max_pairs=max_change_pairs,
            )
        else:
            self.token_projector = ChangeTokenProjector(
                self.encoder.channels[-1], self.text.hidden_size, token_grid
            )

    def prepare_semantic_inputs(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if self.vl_model is None:
            return None
        if not isinstance(self.text, QwenVLAdapter):
            raise TypeError("vl_model is set but the text adapter is not QwenVLAdapter")
        return self.text.prepare_video_inputs(frames, frame_mask)

    @staticmethod
    def prompts(
        domains: Sequence[str],
        gsds: torch.Tensor,
        frame_counts: Sequence[int],
    ) -> list[str]:
        return [
            "你是遥感时序变化分析助手。"
            f"业务={domain}，GSD={float(gsd):.2f}米，时相数={frame_count}。"
            "综合分析整个时序，描述主要变化类型、过程和影响："
            for domain, gsd, frame_count in zip(
                domains, gsds.detach().cpu(), frame_counts, strict=True
            )
        ]

    def encode_sequence(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        gsd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")
        batch, time, channels, height, width = frames.shape
        if time < 2 or frame_mask.shape != (batch, time):
            raise ValueError("frame_mask must match [batch, time] and contain at least two frames")
        if torch.any(frame_mask.sum(dim=1) < 2):
            raise ValueError("each sequence must contain at least two valid frames")

        encoded_flat = self.encoder(frames.reshape(batch * time, channels, height, width))
        encoded = [
            feature.reshape(batch, time, feature.shape[1], feature.shape[2], feature.shape[3])
            for feature in encoded_flat
        ]
        pair_mask = frame_mask[:, :-1] & frame_mask[:, 1:]
        pairs: list[list[torch.Tensor]] = []
        for pair_index in range(time - 1):
            before = [feature[:, pair_index] for feature in encoded]
            after = [feature[:, pair_index + 1] for feature in encoded]
            pairs.append(self.fusion(before, after))
        stacked = [
            torch.stack([pair[level] for pair in pairs], dim=1)
            for level in range(len(self.encoder.channels))
        ]
        fused = self.sequence_aggregator(stacked, pair_mask)
        mask_logits = self.decoder(fused, output_size=(height, width))
        if self.ordered_change_tokens:
            if not isinstance(self.token_projector, OrderedChangeTokenProjector):
                raise TypeError("ordered token mode requires OrderedChangeTokenProjector")
            tokens, token_mask = self.token_projector(stacked[-1], pair_mask, gsd)
        else:
            if not isinstance(self.token_projector, ChangeTokenProjector):
                raise TypeError("aggregate token mode requires ChangeTokenProjector")
            tokens = self.token_projector(fused[-1], gsd)
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return mask_logits, tokens, token_mask

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        gsd: torch.Tensor,
        domains: Sequence[str],
        descriptions: Sequence[str],
        semantic_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        mask_logits, tokens, token_mask = self.encode_sequence(frames, frame_mask, gsd)
        frame_counts = frame_mask.sum(dim=1).detach().cpu().tolist()
        text_loss = self.text.language_loss(
            tokens,
            self.prompts(domains, gsd, frame_counts),
            descriptions,
            visual_mask=token_mask,
            semantic_inputs=semantic_inputs,
        )
        return {
            "mask_logits": mask_logits,
            "text_loss": text_loss,
            "visual_tokens": tokens,
            "visual_token_mask": token_mask,
        }

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        gsd: torch.Tensor,
        domains: Sequence[str],
        generate_text: bool = False,
        semantic_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        mask_logits, tokens, token_mask = self.encode_sequence(frames, frame_mask, gsd)
        descriptions = []
        if generate_text:
            frame_counts = frame_mask.sum(dim=1).detach().cpu().tolist()
            descriptions = self.text.generate(
                tokens,
                self.prompts(domains, gsd, frame_counts),
                visual_mask=token_mask,
                semantic_inputs=semantic_inputs,
            )
        return mask_logits.sigmoid(), descriptions
