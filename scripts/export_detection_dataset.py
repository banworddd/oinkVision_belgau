"""Export a lightweight YOLO-style detection dataset from train annotations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.dataset import load_annotation
from oinkvision.env import get_output_root, load_local_env


LABEL_TO_CLASS_ID = {
    "pig": 0,
    "left_back_leg": 1,
    "right_back_leg": 2,
}


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=output_root / "detector_data")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_rows(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_frame(video_path: str, frame_id: int, output_path: Path) -> bool:
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.index_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    images_dir = args.output_dir / "images"
    labels_dir = args.output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for row in tqdm(rows, desc="Export detector dataset"):
        annotation = load_annotation(row["annotation_path"])
        videos = {camera: row[f"{camera}_video"] for camera in ["top", "right", "left", "rear"]}
        pig_id = row["pig_id"]
        for frame in annotation.get("annotations", []):
            frame_id = int(frame["frame_id"])
            for camera, video_path in videos.items():
                objects = frame.get(camera, [])
                valid_objects = [obj for obj in objects if obj.get("label") in LABEL_TO_CLASS_ID]
                if not valid_objects:
                    continue

                stem = f"{pig_id}_{camera}_{frame_id:05d}"
                image_path = images_dir / f"{stem}.jpg"
                label_path = labels_dir / f"{stem}.txt"

                if not image_path.exists():
                    ok = save_frame(video_path, frame_id, image_path)
                    if not ok:
                        continue

                with label_path.open("w", encoding="utf-8") as f:
                    for obj in valid_objects:
                        cls_id = LABEL_TO_CLASS_ID[obj["label"]]
                        x_center, y_center, width, height = obj["bbox"]
                        f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
                exported += 1

    print(f"Exported {exported} labeled detector frames to {args.output_dir}")


if __name__ == "__main__":
    main()
