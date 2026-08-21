from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PreparedText:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class TinyByteTokenizer:
    """Dependency-free byte tokenizer used only for offline smoke tests."""

    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        values = bytes(item for item in ids if 0 <= item <= 255)
        return values.decode("utf-8", errors="replace")


class TinyCausalLM(nn.Module):
    """Small GRU language head. It validates multimodal gradients without downloads."""

    def __init__(self, hidden_size: int = 192) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(TinyByteTokenizer.vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, TinyByteTokenizer.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        hidden, _ = self.rnn(inputs_embeds)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_from_embeds(self, inputs_embeds: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        hidden_states, hidden = self.rnn(inputs_embeds)
        token = self.lm_head(hidden_states[:, -1]).argmax(dim=-1)
        generated = []
        for _ in range(max_new_tokens):
            generated.append(token)
            embedding = self.embedding(token).unsqueeze(1)
            state, hidden = self.rnn(embedding, hidden)
            token = self.lm_head(state[:, -1]).argmax(dim=-1)
        return torch.stack(generated, dim=1)


class TextAdapter(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_text_tokens: int = 128,
        lora_rank: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_text_tokens = max_text_tokens
        self.is_tiny = model_name == "tiny"

        if self.is_tiny:
            self.tokenizer = TinyByteTokenizer()
            self.model = TinyCausalLM()
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen text model requires: pip install -e '.[qwen]'"
                ) from exc

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self.model.config.use_cache = False
            if lora_rank > 0:
                try:
                    from peft import LoraConfig, TaskType, get_peft_model
                except ImportError as exc:
                    raise RuntimeError("LoRA requires peft: pip install -e '.[qwen]'") from exc
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
                self.model = get_peft_model(self.model, config)

        config = self.model.config
        self.hidden_size = int(
            getattr(config, "hidden_size", getattr(getattr(config, "text_config", None), "hidden_size", 0))
        )
        if not self.hidden_size:
            raise ValueError(f"Cannot determine hidden size for {model_name}")

    def freeze_model(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _encode(self, text: str, add_bos: bool, add_eos: bool) -> list[int]:
        if self.is_tiny:
            return self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        ids = self.tokenizer.encode(text, add_special_tokens=add_bos)
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
        for prompt, target in zip(prompts, targets, strict=True):
            prompt_ids = self._encode(prompt, add_bos=True, add_eos=False)
            # Always reserve at least half the sequence for supervised target tokens.
            # This matters for the byte tokenizer, where one Chinese character uses
            # multiple tokens and a seemingly short prompt can fill the context.
            prompt_ids = prompt_ids[: max(1, self.max_text_tokens // 2)]
            target_ids = self._encode(target, add_bos=False, add_eos=True)
            available = max(1, self.max_text_tokens - len(prompt_ids))
            target_ids = target_ids[:available]
            ids = (prompt_ids + target_ids)[: self.max_text_tokens]
            labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_text_tokens]
            examples.append((ids, labels))

        max_length = max(len(ids) for ids, _ in examples)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((len(examples), max_length), pad_id, dtype=torch.long)
        labels = torch.full((len(examples), max_length), -100, dtype=torch.long)
        attention = torch.zeros((len(examples), max_length), dtype=torch.long)
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

    @staticmethod
    def _prepend_visual_tokens(
        visual_tokens: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # Vision modules normally emit FP32 tensors during inference, while
        # pretrained language models are commonly loaded as BF16. torch.cat
        # promotes the mixed inputs to FP32, which then fails in BF16 linear
        # layers. Match the language embedding dtype explicitly; autocast made
        # this mismatch invisible during training.
        visual_tokens = visual_tokens.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype,
        )
        return torch.cat((visual_tokens, text_embeddings), dim=1)

    def language_loss(
        self,
        visual_tokens: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
        visual_mask: torch.Tensor | None = None,
        semantic_inputs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del semantic_inputs
        prepared = self.prepare_supervision(prompts, targets, visual_tokens.device)
        text_embeddings = self.embed(prepared.input_ids)
        inputs_embeds = self._prepend_visual_tokens(visual_tokens, text_embeddings)
        prefix_length = visual_tokens.shape[1]
        if visual_mask is None:
            prefix_mask = torch.ones(
                (visual_tokens.shape[0], prefix_length),
                dtype=prepared.attention_mask.dtype,
                device=visual_tokens.device,
            )
        else:
            prefix_mask = visual_mask.to(
                device=visual_tokens.device,
                dtype=prepared.attention_mask.dtype,
            )
        attention_mask = torch.cat((prefix_mask, prepared.attention_mask), dim=1)
        prefix_labels = torch.full(
            (visual_tokens.shape[0], prefix_length),
            -100,
            dtype=prepared.labels.dtype,
            device=visual_tokens.device,
        )
        labels = torch.cat((prefix_labels, prepared.labels), dim=1)
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output.loss

    @torch.no_grad()
    def generate(
        self,
        visual_tokens: torch.Tensor,
        prompts: Sequence[str],
        max_new_tokens: int = 48,
        visual_mask: torch.Tensor | None = None,
        semantic_inputs: dict[str, Any] | None = None,
    ) -> list[str]:
        del semantic_inputs
        prompt_ids = [self._encode(prompt, add_bos=True, add_eos=False) for prompt in prompts]
        max_length = max(len(ids) for ids in prompt_ids)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full(
            (len(prompts), max_length), pad_id, dtype=torch.long, device=visual_tokens.device
        )
        attention = torch.zeros_like(input_ids)
        for row, ids in enumerate(prompt_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids, device=visual_tokens.device)
            attention[row, : len(ids)] = 1
        prompt_embeddings = self.embed(input_ids)
        inputs_embeds = self._prepend_visual_tokens(visual_tokens, prompt_embeddings)
        if visual_mask is None:
            prefix_mask = torch.ones(
                visual_tokens.shape[:2], dtype=attention.dtype, device=visual_tokens.device
            )
        else:
            prefix_mask = visual_mask.to(device=visual_tokens.device, dtype=attention.dtype)
        full_attention = torch.cat((prefix_mask, attention), dim=1)

        if self.is_tiny:
            generated = self.model.generate_from_embeds(inputs_embeds, max_new_tokens)
            return [self.tokenizer.decode(row.tolist()) for row in generated]

        self.model.config.use_cache = True
        generated = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model.config.use_cache = False
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
