"""Pre-extract annotated frames into an image cache to speed up training/inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.constants import CAMERAS
from oinkvision.dataset import build_cache_frame_path, load_annotation, load_index
from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=output_root / "frame_cache")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_frame(video_path: str, frame_id: int):
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_id} from {video_path}")
    return frame


def main() -> None:
    args = parse_args()
    rows = load_index(args.index_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    work_items: list[tuple[dict, str, int]] = []
    for row in rows:
        annotation = load_annotation(row["annotation_path"])
        per_camera_frames = {camera: set() for camera in CAMERAS}
        for frame in annotation.get("annotations", []):
            frame_id = int(frame["frame_id"])
            for camera in CAMERAS:
                if frame.get(camera):
                    per_camera_frames[camera].add(frame_id)

        for camera, frame_ids in per_camera_frames.items():
            for frame_id in sorted(frame_ids):
                work_items.append((row, camera, frame_id))

    total_items = len(work_items)
    saved = 0
    skipped = 0
    progress = tqdm(work_items, total=total_items, desc="Preextract frames", unit="frame")
    for row, camera, frame_id in progress:
        cache_path = build_cache_frame_path(args.output_dir, row["pig_id"], camera, frame_id)
        if cache_path.exists():
            skipped += 1
            progress.set_postfix(saved=saved, skipped=skipped)
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame = read_frame(row[f"{camera}_video"], frame_id)
        cv2.imwrite(str(cache_path), frame)
        saved += 1
        progress.set_postfix(saved=saved, skipped=skipped)

    print(f"Saved {saved} cached frames to {args.output_dir}")
    print(f"Skipped {skipped} already cached frames")
    print(f"Total frame tasks: {total_items}")


if __name__ == "__main__":
    main()
