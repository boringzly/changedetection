from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _frame_value(frame: str | dict[str, Any]) -> str:
    if isinstance(frame, str):
        return frame
    value = frame.get("path") or frame.get("image") or frame.get("tif")
    if not value:
        raise ValueError(f"Sequence frame is missing path/image/tif: {frame}")
    return str(value)


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


class SequenceManifestDataset(Dataset[dict[str, Any]]):
    """TIF/imagery sequence dataset without pixel-level change masks.

    Each JSONL record contains at least two time-ordered ``frames``. A frame
    can be a path string or an object such as ``{"path": "2024.tif",
    "timestamp": "2024-06-01"}``. Descriptions are optional for unsupervised
    training and required by the weakly supervised trainer.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        in_channels: int = 3,
        pair_mode: Literal["endpoints", "adjacent", "random"] = "endpoints",
        augment: bool = False,
        require_description: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.base = self.manifest.parent
        self.image_size = image_size
        self.in_channels = in_channels
        self.pair_mode = pair_mode
        self.augment = augment
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        for index, record in enumerate(self.records):
            frames = record.get("frames")
            if not isinstance(frames, Sequence) or isinstance(frames, str) or len(frames) < 2:
                raise ValueError(f"Record {index} must contain at least two frames")
            if require_description and not str(record.get("description", "")).strip():
                raise ValueError(f"Record {index} is missing Claude description")

    def __len__(self) -> int:
        return len(self.records)

    def _pair_indices(self, length: int) -> tuple[int, int]:
        if self.pair_mode == "endpoints":
            return 0, length - 1
        if self.pair_mode == "adjacent":
            first = random.randrange(length - 1)
            return first, first + 1
        if self.pair_mode == "random":
            first, second = sorted(random.sample(range(length), 2))
            return first, second
        raise ValueError(f"Unsupported pair mode: {self.pair_mode}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        frames = record["frames"]
        first_index, second_index = self._pair_indices(len(frames))
        first_path = _resolve(self.base, _frame_value(frames[first_index]))
        second_path = _resolve(self.base, _frame_value(frames[second_index]))
        t1 = _load_image(first_path, self.in_channels, self.image_size)
        t2 = _load_image(second_path, self.in_channels, self.image_size)

        if self.augment:
            if torch.rand(()) < 0.5:
                t1, t2 = (torch.flip(item, dims=(-1,)) for item in (t1, t2))
            if torch.rand(()) < 0.5:
                t1, t2 = (torch.flip(item, dims=(-2,)) for item in (t1, t2))

        return {
            "t1": t1,
            "t2": t2,
            "description": str(record.get("description", "")),
            "domain": str(record.get("domain", "general")),
            "gsd": torch.tensor(float(record.get("gsd", 1.0)), dtype=torch.float32),
            "id": str(record.get("id", index)),
            "frame_count": len(frames),
            "first_frame": first_path.as_posix(),
            "second_frame": second_path.as_posix(),
        }


class SequenceFramesDataset(SequenceManifestDataset):
    """Load a fixed-width representation of the complete temporal sequence.

    Sequences shorter than ``max_frames`` are padded by repeating their last
    frame and accompanied by ``frame_mask``. Longer sequences are sampled at
    evenly spaced indices so the full temporal span is represented. Set
    ``max_frames`` at or above the longest sequence to use every source frame.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        in_channels: int = 3,
        max_frames: int = 8,
        augment: bool = False,
        require_description: bool = True,
    ) -> None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        super().__init__(
            manifest=manifest,
            image_size=image_size,
            in_channels=in_channels,
            pair_mode="endpoints",
            augment=False,
            require_description=require_description,
        )
        self.max_frames = max_frames
        self.sequence_augment = augment

    def _selected_indices(self, length: int) -> list[int]:
        if length <= self.max_frames:
            return list(range(length))
        values = np.linspace(0, length - 1, num=self.max_frames)
        return [int(round(value)) for value in values]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_frames = record["frames"]
        indices = self._selected_indices(len(source_frames))
        tensors = [
            _load_image(
                _resolve(self.base, _frame_value(source_frames[frame_index])),
                self.in_channels,
                self.image_size,
            )
            for frame_index in indices
        ]
        selected_count = len(tensors)
        while len(tensors) < self.max_frames:
            tensors.append(tensors[-1].clone())
        frames = torch.stack(tensors, dim=0)
        frame_mask = torch.zeros(self.max_frames, dtype=torch.bool)
        frame_mask[:selected_count] = True

        if self.sequence_augment:
            if torch.rand(()) < 0.5:
                frames = torch.flip(frames, dims=(-1,))
            if torch.rand(()) < 0.5:
                frames = torch.flip(frames, dims=(-2,))

        return {
            "frames": frames,
            "frame_mask": frame_mask,
            "description": str(record.get("description", "")),
            "domain": str(record.get("domain", "general")),
            "gsd": torch.tensor(float(record.get("gsd", 1.0)), dtype=torch.float32),
            "id": str(record.get("id", index)),
            "frame_count": len(source_frames),
            "selected_frame_count": selected_count,
        }
