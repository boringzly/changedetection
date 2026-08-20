from __future__ import annotations

import json

import torch
from PIL import Image

from geochangemllm.data import SequenceFramesDataset
from geochangemllm.losses import joint_loss
from geochangemllm.losses import pseudo_change_loss, sequence_pseudo_change_targets
from geochangemllm.model import GeoChangeMLLM, GeoSequenceChangeMLLM
from geochangemllm.text import TextAdapter


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


def test_visual_tokens_follow_language_embedding_dtype() -> None:
    visual_tokens = torch.rand(2, 17, 32, dtype=torch.float32)
    text_embeddings = torch.rand(2, 8, 32, dtype=torch.bfloat16)

    combined = TextAdapter._prepend_visual_tokens(visual_tokens, text_embeddings)

    assert combined.dtype == torch.bfloat16
    assert combined.shape == (2, 25, 32)


def test_sequence_dataset_loads_all_frames_and_padding(tmp_path) -> None:
    frame_paths = []
    for index in range(3):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (32, 32), (40 + index * 20, 80, 100)).save(path)
        frame_paths.append(path.name)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "scene-1",
                "frames": frame_paths,
                "description": "水体逐步扩大",
                "domain": "ecology",
                "gsd": 2.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    sample = SequenceFramesDataset(manifest, image_size=32, max_frames=4)[0]

    assert sample["frames"].shape == (4, 3, 32, 32)
    assert sample["frame_mask"].tolist() == [True, True, True, False]
    assert sample["selected_frame_count"] == 3


def test_sequence_weak_supervision_forward_backward() -> None:
    model = GeoSequenceChangeMLLM(base_channels=8, text_model="tiny", token_grid=2)
    frames = torch.rand(2, 3, 3, 64, 64)
    frame_mask = torch.ones(2, 3, dtype=torch.bool)
    gsd = torch.tensor([0.5, 10.0])
    output = model(
        frames,
        frame_mask,
        gsd,
        ["forest", "disaster"],
        ["林地逐步减少", "坡面裸露范围扩大"],
    )
    targets, confidence, _ = sequence_pseudo_change_targets(frames, frame_mask)
    pseudo, _ = pseudo_change_loss(output["mask_logits"], targets, confidence)
    loss = pseudo + 0.2 * output["text_loss"]
    loss.backward()

    assert output["mask_logits"].shape == (2, 1, 64, 64)
    assert model.sequence_aggregator.projections[0].weight.grad is not None
    assert model.token_projector.project[1].weight.grad is not None


def test_constant_sequence_is_pseudo_labeled_as_stable() -> None:
    frames = torch.ones(1, 3, 3, 16, 16)
    frame_mask = torch.ones(1, 3, dtype=torch.bool)

    targets, confidence, score = sequence_pseudo_change_targets(frames, frame_mask)

    assert torch.count_nonzero(score) == 0
    assert torch.count_nonzero(targets) == 0
    assert torch.all(confidence == 1)
