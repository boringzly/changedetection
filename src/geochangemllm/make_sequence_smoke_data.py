from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DOMAINS = ["forest", "disaster", "natural_resources", "ecology"]
DESCRIPTIONS = {
    "forest": "前两期林地基本稳定，随后中心区域持续被清理，末期形成明显裸地斑块",
    "disaster": "序列中期坡面开始出现裸露，后续范围扩大，表现为疑似渐进式滑坡",
    "natural_resources": "前期地表未见明显变化，中后期出现并逐步扩张为规则建设用地区域",
    "ecology": "时序影像显示局部水体从无到有并逐期扩张，末期形成连续水面",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create georeferenced TIF sequence smoke data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sequence-smoke"))
    parser.add_argument("--num-sequences", type=int, default=8)
    parser.add_argument("--frames-per-sequence", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def textured_background(size: int, color: tuple[int, int, int], rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(image)
    for _ in range(120):
        x, y = rng.randrange(size), rng.randrange(size)
        radius = rng.randint(1, 4)
        value = tuple(max(0, min(255, channel + rng.randint(-30, 30))) for channel in color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
    return image


def add_progressive_change(
    image: Image.Image,
    domain: str,
    progress: float,
    box: tuple[int, int, int, int],
) -> None:
    if progress <= 0:
        return
    x0, y0, x1, y1 = box
    current_x1 = int(x0 + (x1 - x0) * progress)
    current_y1 = int(y0 + (y1 - y0) * progress)
    draw = ImageDraw.Draw(image)
    if domain == "forest":
        draw.rectangle((x0, y0, current_x1, current_y1), fill=(155, 122, 82))
    elif domain == "disaster":
        polygon = [(x0, y0), (current_x1, y0 + 8), (current_x1 - 10, current_y1), (x0 + 8, current_y1)]
        draw.polygon(polygon, fill=(142, 96, 62))
    elif domain == "natural_resources":
        draw.rectangle((x0, y0, current_x1, current_y1), fill=(178, 178, 174))
    else:
        draw.ellipse((x0, y0, current_x1, current_y1), fill=(50, 112, 158))


def save_geotiff(path: Path, image: Image.Image, gsd: float) -> None:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise RuntimeError("Sequence smoke generation requires rasterio") from exc
    array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(500000.0, 4000000.0, gsd, gsd),
        compress="deflate",
    ) as dst:
        dst.write(array)


def main() -> None:
    args = parse_args()
    if args.frames_per_sequence < 2:
        raise ValueError("frames-per-sequence must be at least 2")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    records = []
    colors = [(58, 104, 58), (104, 112, 74), (86, 108, 116), (106, 114, 88)]
    for index in range(args.num_sequences):
        domain = DOMAINS[index % len(DOMAINS)]
        sequence_id = f"sequence_{index:04d}"
        sequence_dir = root / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=True)
        gsd = round(rng.uniform(0.3, 10.0), 2)
        base = textured_background(args.image_size, colors[index % len(colors)], rng)
        margin = max(12, args.image_size // 8)
        box = (
            rng.randint(margin, args.image_size // 3),
            rng.randint(margin, args.image_size // 3),
            rng.randint(args.image_size * 2 // 3, args.image_size - margin),
            rng.randint(args.image_size * 2 // 3, args.image_size - margin),
        )
        frame_paths = []
        for frame_index in range(args.frames_per_sequence):
            frame = base.copy()
            progress = frame_index / (args.frames_per_sequence - 1)
            progress = max(0.0, (progress - 0.20) / 0.80)
            add_progressive_change(frame, domain, progress, box)
            frame_path = sequence_dir / f"2024{frame_index + 1:02d}01.tif"
            save_geotiff(frame_path, frame, gsd)
            frame_paths.append(frame_path.relative_to(root).as_posix())
        records.append(
            {
                "id": sequence_id,
                "frames": frame_paths,
                "description": DESCRIPTIONS[domain],
                "domain": domain,
                "gsd": gsd,
            }
        )
    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"created {len(records)} TIF sequences: {manifest}")


if __name__ == "__main__":
    main()
