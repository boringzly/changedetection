from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from .text import PreparedText


class QwenVLAdapter(nn.Module):
    """Qwen3-VL adapter that joins native video and ordered change tokens.

    Qwen's native vision encoder supplies semantic video tokens. A separate
    remote-sensing change branch supplies ordered adjacent-pair tokens. Both
    prefixes are concatenated before the language prompt, while only the
    Claude target tokens participate in the causal language-model loss.
    """

    def __init__(
        self,
        model_name: str,
        max_text_tokens: int = 256,
        lora_rank: int = 16,
        freeze_vision: bool = True,
        min_pixels: int = 50_176,
        max_pixels: int = 200_704,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL requires transformers with Qwen3VL support: "
                "pip install -e '.[qwen]'"
            ) from exc

        self.model_name = model_name
        self.max_text_tokens = max_text_tokens
        self.freeze_vision = freeze_vision
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        self.hidden_size = int(model.config.text_config.hidden_size)

        if freeze_vision:
            for parameter in model.model.visual.parameters():
                parameter.requires_grad = False

        if lora_rank > 0:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise RuntimeError("Qwen3-VL LoRA requires peft: pip install -e '.[qwen]'") from exc
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            model = get_peft_model(model, config)
        if freeze_vision:
            raw_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            for parameter in raw_model.model.visual.parameters():
                parameter.requires_grad = False
        self.model = model

        # Modality embeddings make the two visual streams distinguishable.
        # The gate starts small so training begins close to native Qwen3-VL.
        self.semantic_type = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.change_type = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.change_gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.semantic_type, std=0.02)
        nn.init.normal_(self.change_type, std=0.02)

    def freeze_model(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _raw_model(self) -> nn.Module:
        if hasattr(self.model, "get_base_model"):
            return self.model.get_base_model()
        return self.model

    def prepare_video_inputs(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the official Qwen processor on a CPU batch of [B,T,C,H,W]."""
        if frames.device.type != "cpu" or frame_mask.device.type != "cpu":
            raise ValueError("prepare_video_inputs expects CPU tensors before device transfer")
        if frames.ndim != 5 or frame_mask.shape != frames.shape[:2]:
            raise ValueError("frames/frame_mask must have shapes [B,T,C,H,W] and [B,T]")
        if frames.shape[2] < 3:
            raise ValueError("Qwen3-VL semantic input requires at least three display bands")

        videos: list[list[Image.Image]] = []
        for sample_frames, valid in zip(frames, frame_mask, strict=True):
            valid_frames = sample_frames[valid]
            if valid_frames.shape[0] < 2:
                raise ValueError("Qwen3-VL sequence input requires at least two valid frames")
            images = []
            for frame in valid_frames[:, :3]:
                array = (
                    frame.clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .numpy()
                )
                images.append(Image.fromarray(np.asarray(array), mode="RGB"))
            videos.append(images)

        encoded = self.processor(videos=videos, return_tensors="pt")
        required = ("pixel_values_videos", "video_grid_thw")
        missing = [name for name in required if name not in encoded]
        if missing:
            raise RuntimeError(f"Qwen3-VL processor did not return: {missing}")
        return {name: encoded[name] for name in required}

    def _format_prompts(self, prompts: Sequence[str]) -> list[str]:
        formatted = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return formatted

    def _encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_eos and self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return ids

    def prepare_supervision(
        self,
        prompts: Sequence[str],
        targets: Sequence[str],
        device: torch.device,
    ) -> PreparedText:
        examples: list[tuple[list[int], list[int]]] = []
        for prompt, target in zip(self._format_prompts(prompts), targets, strict=True):
            prompt_ids = self._encode(prompt)[: max(1, self.max_text_tokens // 2)]
            target_ids = self._encode(target, add_eos=True)
            target_ids = target_ids[: max(1, self.max_text_tokens - len(prompt_ids))]
            ids = (prompt_ids + target_ids)[: self.max_text_tokens]
            labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_text_tokens]
            examples.append((ids, labels))

        max_length = max(len(ids) for ids, _ in examples)
        input_ids = torch.full(
            (len(examples), max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
        )
        labels = torch.full_like(input_ids, -100)
        attention = torch.zeros_like(input_ids)
        for row, (ids, row_labels) in enumerate(examples):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            labels[row, :length] = torch.tensor(row_labels, dtype=torch.long)
            attention[row, :length] = 1
        return PreparedText(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
            labels=labels.to(device),
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def encode_semantic_tokens(
        self,
        semantic_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_model = self._raw_model()
        context = torch.no_grad() if self.freeze_vision else nullcontext()
        with context:
            output = raw_model.get_video_features(
                pixel_values_videos=semantic_inputs["pixel_values_videos"],
                video_grid_thw=semantic_inputs["video_grid_thw"],
                return_dict=True,
            )
        features = output.pooler_output
        if isinstance(features, torch.Tensor):
            features = (features,)
        if not features:
            raise RuntimeError("Qwen3-VL vision encoder returned no video features")

        max_tokens = max(feature.shape[0] for feature in features)
        tokens = features[0].new_zeros((len(features), max_tokens, self.hidden_size))
        mask = torch.zeros((len(features), max_tokens), dtype=torch.bool, device=tokens.device)
        for row, feature in enumerate(features):
            length = feature.shape[0]
            tokens[row, :length] = feature
            mask[row, :length] = True
        return tokens, mask

    def combine_visual_prefixes(
        self,
        semantic_tokens: torch.Tensor,
        semantic_mask: torch.Tensor,
        change_tokens: torch.Tensor,
        change_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = semantic_tokens.dtype
        device = semantic_tokens.device
        semantic = semantic_tokens + self.semantic_type.to(device=device, dtype=dtype)
        change = change_tokens.to(device=device, dtype=dtype)
        change = torch.sigmoid(self.change_gate).to(dtype) * change
        change = change + self.change_type.to(device=device, dtype=dtype)
        tokens = torch.cat((semantic, change), dim=1)
        mask = torch.cat((semantic_mask, change_mask.to(device=device)), dim=1)
        return tokens, mask

    def _build_inputs(
        self,
        change_tokens: torch.Tensor,
        change_mask: torch.Tensor,
        semantic_inputs: dict[str, torch.Tensor],
        prompts: Sequence[str],
        targets: Sequence[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        semantic_tokens, semantic_mask = self.encode_semantic_tokens(semantic_inputs)
        visual_tokens, visual_mask = self.combine_visual_prefixes(
            semantic_tokens,
            semantic_mask,
            change_tokens,
            change_mask,
        )
        if targets is None:
            formatted = self._format_prompts(prompts)
            ids = [self._encode(prompt) for prompt in formatted]
            max_length = max(len(row) for row in ids)
            input_ids = torch.full(
                (len(ids), max_length),
                int(self.tokenizer.pad_token_id),
                dtype=torch.long,
                device=visual_tokens.device,
            )
            text_mask = torch.zeros_like(input_ids)
            for row, values in enumerate(ids):
                input_ids[row, : len(values)] = torch.tensor(values, device=input_ids.device)
                text_mask[row, : len(values)] = 1
            labels = None
        else:
            prepared = self.prepare_supervision(prompts, targets, visual_tokens.device)
            input_ids = prepared.input_ids
            text_mask = prepared.attention_mask
            labels = prepared.labels

        text_embeddings = self.embed(input_ids)
        visual_tokens = visual_tokens.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        inputs_embeds = torch.cat((visual_tokens, text_embeddings), dim=1)
        attention_mask = torch.cat((visual_mask.to(text_mask.dtype), text_mask), dim=1)
        if labels is not None:
            prefix_labels = torch.full(
                visual_mask.shape,
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat((prefix_labels, labels), dim=1)
        return inputs_embeds, attention_mask, labels

    def language_loss(
        self,
        visual_tokens: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
        visual_mask: torch.Tensor | None = None,
        semantic_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if semantic_inputs is None:
            raise ValueError("Qwen3-VL language loss requires native video semantic inputs")
        if visual_mask is None:
            visual_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
        inputs_embeds, attention_mask, labels = self._build_inputs(
            visual_tokens,
            visual_mask,
            semantic_inputs,
            prompts,
            targets,
        )
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return output.loss

    @torch.no_grad()
    def generate(
        self,
        visual_tokens: torch.Tensor,
        prompts: Sequence[str],
        max_new_tokens: int = 96,
        visual_mask: torch.Tensor | None = None,
        semantic_inputs: dict[str, torch.Tensor] | None = None,
    ) -> list[str]:
        if semantic_inputs is None:
            raise ValueError("Qwen3-VL generation requires native video semantic inputs")
        if visual_mask is None:
            visual_mask = torch.ones(visual_tokens.shape[:2], dtype=torch.bool, device=visual_tokens.device)
        inputs_embeds, attention_mask, _ = self._build_inputs(
            visual_tokens,
            visual_mask,
            semantic_inputs,
            prompts,
            targets=None,
        )
        self.model.config.use_cache = True
        generated = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.10,
            no_repeat_ngram_size=4,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model.config.use_cache = False
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
