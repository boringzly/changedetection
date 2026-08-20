from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _scale_uint8(array: np.ndarray) -> np.ndarray:
    """Robust per-channel scaling for non-byte remote-sensing rasters."""
    array = array.astype(np.float32, copy=False)
    result = np.zeros_like(array, dtype=np.float32)
    for channel in range(array.shape[0]):
        band = array[channel]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        low, high = np.percentile(valid, (2, 98))
        if high <= low:
            high = low + 1.0
        result[channel] = np.clip((band - low) / (high - low), 0.0, 1.0)
    return result


def _load_image(path: Path, channels: int, size: int) -> torch.Tensor:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError(
                "GeoTIFF requires rasterio. Install with: pip install -e '.[geo]'"
            ) from exc
        with rasterio.open(path) as src:
            count = min(channels, src.count)
            array = src.read(list(range(1, count + 1)))
        array = _scale_uint8(array)
        tensor = torch.from_numpy(array)
    else:
        image = Image.open(path).convert("RGB")
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)

    if tensor.shape[0] < channels:
        padding = tensor.new_zeros((channels - tensor.shape[0], *tensor.shape[1:]))
        tensor = torch.cat((tensor, padding), dim=0)
    tensor = tensor[:channels].unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)
    return tensor.clamp(0.0, 1.0)


def _load_mask(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = image.resize((size, size), Image.Resampling.NEAREST)
    array = (np.asarray(image, dtype=np.uint8) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class ChangeManifestDataset(Dataset[dict[str, Any]]):
    """JSONL dataset shared by smoke data and future real GeoTIFF data."""

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        in_channels: int = 3,
        augment: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.base = self.manifest.parent
        self.image_size = image_size
        self.in_channels = in_channels
        self.augment = augment
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        t1 = _load_image(
            _resolve(self.base, record["t1"]), self.in_channels, self.image_size
        )
        t2 = _load_image(
            _resolve(self.base, record["t2"]), self.in_channels, self.image_size
        )
        mask = _load_mask(_resolve(self.base, record["mask"]), self.image_size)

        if self.augment:
            if torch.rand(()) < 0.5:
                t1, t2, mask = (torch.flip(item, dims=(-1,)) for item in (t1, t2, mask))
            if torch.rand(()) < 0.5:
                t1, t2, mask = (torch.flip(item, dims=(-2,)) for item in (t1, t2, mask))

        return {
            "t1": t1,
            "t2": t2,
            "mask": mask,
            "description": str(record.get("description", "未提供变化描述")),
            "domain": str(record.get("domain", "general")),
            "gsd": torch.tensor(float(record.get("gsd", 1.0)), dtype=torch.float32),
            "id": str(record.get("id", index)),
        }
