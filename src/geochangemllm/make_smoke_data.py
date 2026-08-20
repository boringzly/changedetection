from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


DOMAINS = ["forest", "disaster", "natural_resources", "ecology"]
COLORS = [(55, 105, 55), (95, 120, 70), (80, 105, 120), (120, 110, 85)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/smoke"))
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def add_texture(image: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(80):
        x = rng.randrange(width)
        y = rng.randrange(height)
        radius = rng.randint(1, 5)
        color = tuple(rng.randint(35, 150) for _ in range(3))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def generate_sample(root: Path, index: int, size: int, rng: random.Random) -> dict[str, object]:
    domain = DOMAINS[index % len(DOMAINS)]
    background = COLORS[index % len(COLORS)]
    t1 = Image.new("RGB", (size, size), background)
    add_texture(t1, rng)
    t2 = t1.copy()
    mask = Image.new("L", (size, size), 0)

    margin = max(8, size // 12)
    x0 = rng.randint(margin, size // 2)
    y0 = rng.randint(margin, size // 2)
    x1 = rng.randint(size // 2, size - margin)
    y1 = rng.randint(size // 2, size - margin)

    after_draw = ImageDraw.Draw(t2)
    mask_draw = ImageDraw.Draw(mask)
    if domain == "forest":
        after_draw.rectangle((x0, y0, x1, y1), fill=(150, 120, 85))
        mask_draw.rectangle((x0, y0, x1, y1), fill=255)
        description = "中心区域林地被清理为裸地"
    elif domain == "disaster":
        polygon = [(x0, y0), (x1, y0 + 10), (x1 - 15, y1), (x0 + 10, y1)]
        after_draw.polygon(polygon, fill=(135, 95, 65))
        mask_draw.polygon(polygon, fill=255)
        description = "坡面出现疑似滑坡裸露区域"
    elif domain == "natural_resources":
        after_draw.rectangle((x0, y0, x1, y1), fill=(175, 175, 170))
        mask_draw.rectangle((x0, y0, x1, y1), fill=255)
        description = "新增规则建设用地区域"
    else:
        after_draw.ellipse((x0, y0, x1, y1), fill=(55, 110, 150))
        mask_draw.ellipse((x0, y0, x1, y1), fill=255)
        description = "局部新增水体覆盖"

    stem = f"sample_{index:04d}"
    t1_path = root / "images" / f"{stem}_t1.png"
    t2_path = root / "images" / f"{stem}_t2.png"
    mask_path = root / "masks" / f"{stem}.png"
    t1.save(t1_path)
    t2.save(t2_path)
    mask.save(mask_path)
    return {
        "id": stem,
        "t1": t1_path.relative_to(root).as_posix(),
        "t2": t2_path.relative_to(root).as_posix(),
        "mask": mask_path.relative_to(root).as_posix(),
        "description": description,
        "domain": domain,
        "gsd": round(rng.uniform(0.3, 10.0), 2),
    }


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records = [generate_sample(root, i, args.image_size, rng) for i in range(args.num_samples)]
    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"created {len(records)} samples: {manifest}")


if __name__ == "__main__":
    main()
