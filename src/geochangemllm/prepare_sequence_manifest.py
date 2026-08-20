from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TIF sequence JSONL manifest")
    parser.add_argument("--tif-root", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-domain", default="general")
    parser.add_argument("--default-gsd", type=float, default=1.0)
    return parser.parse_args()


def load_descriptions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("id", "")).strip()
            if not record_id:
                raise ValueError(f"Description line {line_number} is missing id")
            if "description" not in record:
                record["description"] = record.get("text", record.get("caption", ""))
            records[record_id] = record
    return records


def tif_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def main() -> None:
    args = parse_args()
    root = args.tif_root.resolve()
    output = args.output.resolve()
    descriptions = load_descriptions(args.descriptions.resolve() if args.descriptions else None)
    direct_frames = tif_files(root)
    if direct_frames:
        sequence_directories = [(root.name, root, direct_frames)]
    else:
        sequence_directories = [
            (directory.name, directory, frames)
            for directory in sorted(path for path in root.iterdir() if path.is_dir())
            if (frames := tif_files(directory))
        ]
    if not sequence_directories:
        raise ValueError(f"No TIF sequences found under {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sequence_id, _, frames in sequence_directories:
            if len(frames) < 2:
                raise ValueError(f"Sequence {sequence_id} has fewer than two TIF frames")
            metadata = descriptions.get(sequence_id, {})
            relative_frames = [
                Path(os.path.relpath(frame, output.parent)).as_posix() for frame in frames
            ]
            record = {
                "id": sequence_id,
                "frames": relative_frames,
                "description": str(metadata.get("description", "")),
                "domain": str(metadata.get("domain", args.default_domain)),
                "gsd": float(metadata.get("gsd", args.default_gsd)),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    missing = [sequence_id for sequence_id, _, _ in sequence_directories if sequence_id not in descriptions]
    print(f"created {len(sequence_directories)} sequences: {output}")
    if args.descriptions and missing:
        print(f"warning: {len(missing)} sequences have no Claude description: {', '.join(missing[:5])}")


if __name__ == "__main__":
    main()
