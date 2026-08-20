from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import SequenceManifestDataset
from .losses import (
    pseudo_change_loss,
    pseudo_change_targets,
    stable_feature_consistency,
    temporal_symmetry_loss,
)
from .vision import VisionChangeModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised TIF sequence pretraining")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unsupervised"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pair-mode", choices=["adjacent", "random", "endpoints"], default="adjacent")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--change-quantile", type=float, default=0.85)
    parser.add_argument("--stable-quantile", type=float, default=0.50)
    parser.add_argument("--feature-weight", type=float, default=0.10)
    parser.add_argument("--symmetry-weight", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world_size


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def main() -> None:
    args = parse_args()
    if args.feature_weight < 0 or args.symmetry_weight < 0:
        raise ValueError("loss weights must be non-negative")
    distributed, rank, local_rank, world_size = setup_distributed()
    seed_everything(args.seed, rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dataset = SequenceManifestDataset(
        args.manifest,
        image_size=args.image_size,
        in_channels=args.in_channels,
        pair_mode=args.pair_mode,
        augment=True,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=len(dataset) >= args.batch_size * world_size,
    )
    model: torch.nn.Module = VisionChangeModel(args.in_channels, args.base_channels).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model.load_state_dict(checkpoint["vision"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output_dir / "run_config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"mode=unsupervised device={device} world_size={world_size} "
            f"sequences={len(dataset)} pair_mode={args.pair_mode}"
        )

    use_amp = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.precision == "fp16")
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    micro_step = 0
    stop = False
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            t1 = batch["t1"].to(device, non_blocking=True)
            t2 = batch["t2"].to(device, non_blocking=True)
            targets, confidence, score = pseudo_change_targets(
                t1,
                t2,
                change_quantile=args.change_quantile,
                stable_quantile=args.stable_quantile,
            )
            sync_step = (micro_step + 1) % args.grad_accum == 0
            sync_context = nullcontext()
            if distributed and not sync_step:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    forward = model(t1, t2)
                    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                    with torch.no_grad():
                        reverse = raw_model(t2, t1)
                    pseudo, parts = pseudo_change_loss(
                        forward["mask_logits"], targets, confidence
                    )
                    feature = stable_feature_consistency(
                        forward["before_deep"],
                        forward["after_deep"],
                        score,
                        stable_quantile=args.stable_quantile,
                    )
                    symmetry = temporal_symmetry_loss(
                        forward["mask_logits"], reverse["mask_logits"]
                    )
                    loss = pseudo + args.feature_weight * feature + args.symmetry_weight * symmetry
                    scaled_loss = loss / args.grad_accum
                scaler.scale(scaled_loss).backward()

            micro_step += 1
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and global_step % args.log_every == 0:
                pseudo_fraction = targets.mean().item()
                memory = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                print(
                    f"step={global_step} loss={loss.item():.4f} "
                    f"pseudo_bce={parts['pseudo_bce'].item():.4f} "
                    f"pseudo_dice={parts['pseudo_dice'].item():.4f} "
                    f"feature={feature.item():.4f} symmetry={symmetry.item():.4f} "
                    f"pseudo_fraction={pseudo_fraction:.4f} max_mem={memory:.2f}GiB"
                )
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if distributed:
        dist.barrier()
    if rank == 0:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        checkpoint_path = args.output_dir / "vision_checkpoint.pt"
        torch.save(
            {
                "vision": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"saved: {checkpoint_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
