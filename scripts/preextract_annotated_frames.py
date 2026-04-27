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


def collect_work_items(rows: list[dict], output_dir: Path) -> list[tuple[dict, str, str, list[int]]]:
    grouped: list[tuple[dict, str, str, list[int]]] = []

    for row in rows:
        annotation = load_annotation(row["annotation_path"])
        per_camera_frames = {camera: set() for camera in CAMERAS}
        for frame in annotation.get("annotations", []):
            frame_id = int(frame["frame_id"])
            for camera in CAMERAS:
                if frame.get(camera):
                    cache_path = build_cache_frame_path(output_dir, row["pig_id"], camera, frame_id)
                    if not cache_path.exists():
                        per_camera_frames[camera].add(frame_id)

        for camera, frame_ids in per_camera_frames.items():
            if frame_ids:
                grouped.append((row, camera, row[f"{camera}_video"], sorted(frame_ids)))

    return grouped


def extract_frames_for_video(
    video_path: str,
    target_frame_ids: list[int],
    pig_id: str,
    camera: str,
    output_dir: Path,
    progress: tqdm,
) -> int:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved = 0
    current_idx = 0
    target_pos = 0
    targets_set = set(target_frame_ids)
    last_target = target_frame_ids[-1]

    while target_pos < len(target_frame_ids):
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        if current_idx in targets_set:
            cache_path = build_cache_frame_path(output_dir, pig_id, camera, current_idx)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_path), frame)
            saved += 1
            progress.update(1)
            progress.saved_count += 1
            progress.set_postfix(saved=progress.saved_count, skipped=progress.skipped_count)
            target_pos += 1

        if current_idx >= last_target:
            break
        current_idx += 1

    capture.release()
    return saved


def main() -> None:
    args = parse_args()
    rows = load_index(args.index_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    work_groups = collect_work_items(rows, output_dir)
    total_items = sum(len(frame_ids) for _, _, _, frame_ids in work_groups)
    saved = 0
    skipped = 0
    total_requested = 0
    for row in rows:
        annotation = load_annotation(row["annotation_path"])
        for frame in annotation.get("annotations", []):
            for camera in CAMERAS:
                if frame.get(camera):
                    total_requested += 1
    skipped = total_requested - total_items

    progress = tqdm(total=total_items, desc="Preextract frames", unit="frame")
    progress.saved_count = saved
    progress.skipped_count = skipped
    progress.set_postfix(saved=saved, skipped=skipped)

    for row, camera, video_path, frame_ids in work_groups:
        saved_now = extract_frames_for_video(
            video_path=video_path,
            target_frame_ids=frame_ids,
            pig_id=row["pig_id"],
            camera=camera,
            output_dir=output_dir,
            progress=progress,
        )
        saved += saved_now
        progress.skipped_count = skipped
        progress.set_postfix(saved=saved, skipped=skipped)

    progress.close()

    print(f"Saved {saved} cached frames to {args.output_dir}")
    print(f"Skipped {skipped} already cached frames")
    print(f"Total uncached frame tasks: {total_items}")


if __name__ == "__main__":
    main()
