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

from .data import SequenceFramesDataset
from .losses import pseudo_change_loss, sequence_pseudo_change_targets
from .model import GeoSequenceChangeMLLM
from .vision import load_vision_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude-description weak supervision for TIF sequences")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/weak-qwen"))
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--token-grid", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--text-loss-weight", type=float, default=0.20)
    parser.add_argument("--pseudo-loss-weight", type=float, default=1.0)
    parser.add_argument("--change-quantile", type=float, default=0.85)
    parser.add_argument("--stable-quantile", type=float, default=0.50)
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
    if args.pseudo_loss_weight <= 0:
        raise ValueError("pseudo-loss-weight must be positive so the change decoder is supervised")
    if args.text_loss_weight <= 0:
        raise ValueError("text-loss-weight must be positive for Claude weak supervision")
    distributed, rank, local_rank, world_size = setup_distributed()
    seed_everything(args.seed, rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dataset = SequenceFramesDataset(
        args.manifest,
        image_size=args.image_size,
        in_channels=args.in_channels,
        max_frames=args.max_frames,
        augment=True,
        require_description=True,
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
    model: torch.nn.Module = GeoSequenceChangeMLLM(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        text_model=args.text_model,
        lora_rank=args.lora_rank,
        freeze_text=args.freeze_text,
        token_grid=args.token_grid,
        max_text_tokens=args.max_text_tokens,
    )
    if args.vision_checkpoint and not args.resume:
        loaded = load_vision_checkpoint(model, args.vision_checkpoint)
        if rank == 0:
            print(f"loaded_vision_tensors={loaded} from={args.vision_checkpoint}")
    model.to(device)
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
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output_dir / "run_config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        trainable = sum(parameter.numel() for parameter in parameters)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"mode=weak_sequence device={device} world_size={world_size} "
            f"sequences={len(dataset)} max_frames={args.max_frames} "
            f"trainable={trainable:,}/{total:,}"
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
            frames = batch["frames"].to(device, non_blocking=True)
            frame_mask = batch["frame_mask"].to(device, non_blocking=True)
            gsd = batch["gsd"].to(device, non_blocking=True)
            targets, confidence, _ = sequence_pseudo_change_targets(
                frames,
                frame_mask,
                change_quantile=args.change_quantile,
                stable_quantile=args.stable_quantile,
            )
            sync_step = (micro_step + 1) % args.grad_accum == 0
            sync_context = nullcontext()
            if distributed and not sync_step:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output = model(
                        frames,
                        frame_mask,
                        gsd,
                        batch["domain"],
                        batch["description"],
                    )
                    pseudo, parts = pseudo_change_loss(
                        output["mask_logits"], targets, confidence
                    )
                    loss = (
                        args.pseudo_loss_weight * pseudo
                        + args.text_loss_weight * output["text_loss"]
                    )
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
                memory = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                print(
                    f"step={global_step} loss={loss.item():.4f} "
                    f"pseudo_bce={parts['pseudo_bce'].item():.4f} "
                    f"pseudo_dice={parts['pseudo_dice'].item():.4f} "
                    f"text={output['text_loss'].item():.4f} "
                    f"pseudo_fraction={targets.mean().item():.4f} max_mem={memory:.2f}GiB"
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
        checkpoint_path = args.output_dir / "checkpoint.pt"
        torch.save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "args": vars(args),
                "model_type": "sequence",
            },
            checkpoint_path,
        )
        print(f"saved: {checkpoint_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
