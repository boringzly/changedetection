from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import ChangeManifestDataset
from .model import GeoChangeMLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/infer"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--generate-text", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GeoChangeMLLM(
        in_channels=int(train_args["in_channels"]),
        base_channels=int(train_args["base_channels"]),
        text_model=str(train_args["text_model"]),
        lora_rank=int(train_args["lora_rank"]),
        freeze_text=bool(train_args["freeze_text"]),
        token_grid=int(train_args["token_grid"]),
        max_text_tokens=int(train_args["max_text_tokens"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    dataset = ChangeManifestDataset(
        args.manifest,
        image_size=int(train_args["image_size"]),
        in_channels=int(train_args["in_channels"]),
    )
    sample = dataset[args.sample_index]
    t1 = sample["t1"].unsqueeze(0).to(device)
    t2 = sample["t2"].unsqueeze(0).to(device)
    gsd = sample["gsd"].view(1).to(device)
    probability, descriptions = model.predict(
        t1,
        t2,
        gsd,
        [sample["domain"]],
        generate_text=args.generate_text,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    array = (probability[0, 0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(args.output_dir / "change_probability.png")
    result = {
        "id": sample["id"],
        "domain": sample["domain"],
        "gsd": float(sample["gsd"]),
        "threshold": args.threshold,
        "changed_fraction": float((probability >= args.threshold).float().mean().cpu()),
        "generated_description": descriptions[0] if descriptions else None,
        "reference_description": sample["description"],
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
