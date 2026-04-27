"""Dataset for multi-view pig condition classification."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CAMERAS, LABELS


@dataclass
class FrameSample:
    frame_id: int
    camera: str
    bboxes: list[dict[str, Any]]
    source: str = "annotation"


def load_index(index_path: str | Path) -> list[dict[str, Any]]:
    index_path = Path(index_path)
    with index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    int_fields = [
        "bad_posture",
        "bumps",
        "soft_pastern",
        "x_shape",
        "has_target",
        "num_annotated_frames",
        "annotated_top_frames",
        "annotated_right_frames",
        "annotated_left_frames",
        "annotated_rear_frames",
    ]
    for row in rows:
        for field in int_fields:
            row[field] = int(row.get(field, 0) or 0)
    return rows


def load_annotation(annotation_path: str | Path) -> dict[str, Any]:
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cache_frame_path(
    frame_cache_dir: str | Path,
    pig_id: str,
    camera: str,
    frame_id: int,
) -> Path:
    cache_dir = Path(frame_cache_dir)
    return cache_dir / pig_id / camera / f"frame_{frame_id:05d}.jpg"


def get_target_vector(row: dict[str, Any]) -> list[int]:
    return [int(row.get(label, 0)) for label in LABELS]


def collect_frame_samples(
    annotation: dict[str, Any],
    frames_per_camera: int,
    seed: int | None = None,
) -> list[FrameSample]:
    rng = random.Random(seed)
    per_camera: dict[str, list[FrameSample]] = {camera: [] for camera in CAMERAS}

    for frame in annotation.get("annotations", []):
        frame_id = int(frame["frame_id"])
        for camera in CAMERAS:
            bboxes = frame.get(camera, [])
            if not bboxes:
                continue
            per_camera[camera].append(
                FrameSample(
                    frame_id=frame_id,
                    camera=camera,
                    bboxes=bboxes,
                )
            )

    sampled: list[FrameSample] = []
    for camera in CAMERAS:
        candidates = per_camera[camera]
        if not candidates:
            continue

        if len(candidates) <= frames_per_camera:
            selected = sorted(candidates, key=lambda item: item.frame_id)
        else:
            selected = sorted(rng.sample(candidates, frames_per_camera), key=lambda item: item.frame_id)

        sampled.extend(selected)

    return sampled


def get_video_num_frames(video_path: str | Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return max(num_frames, 1)


def collect_uniform_frame_samples(
    video_path: str | Path,
    camera: str,
    frames_per_camera: int,
) -> list[FrameSample]:
    num_frames = get_video_num_frames(video_path)
    if frames_per_camera <= 1:
        frame_ids = [max(num_frames // 2, 0)]
    else:
        frame_ids = np.linspace(0, max(num_frames - 1, 0), num=frames_per_camera, dtype=int).tolist()
    return [
        FrameSample(
            frame_id=int(frame_id),
            camera=camera,
            bboxes=[],
            source="uniform",
        )
        for frame_id in frame_ids
    ]


def bbox_yolo_to_xyxy(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x_center, y_center, width, height = bbox
    box_w = width * image_width
    box_h = height * image_height
    cx = x_center * image_width
    cy = y_center * image_height

    x1 = max(int(cx - box_w / 2), 0)
    y1 = max(int(cy - box_h / 2), 0)
    x2 = min(int(cx + box_w / 2), image_width)
    y2 = min(int(cy + box_h / 2), image_height)
    return x1, y1, x2, y2


class PigVideoDataset(Dataset):
    """Animal-level dataset with fixed-size multi-view frame sampling."""

    def __init__(
        self,
        index_path: str | Path,
        frames_per_camera: int = 8,
        seed: int = 42,
        image_size: int = 224,
        use_bbox_crops: bool = True,
        frame_cache_dir: str | Path | None = None,
        augment: bool = False,
        raw_sample_ratio: float = 0.0,
        augmentation_profile: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.rows = rows if rows is not None else load_index(index_path)
        self.frames_per_camera = frames_per_camera
        self.seed = seed
        self.image_size = image_size
        self.use_bbox_crops = use_bbox_crops
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.augment = augment
        self.raw_sample_ratio = float(np.clip(raw_sample_ratio, 0.0, 1.0))
        self.augmentation_profile = dict(augmentation_profile or {})
        self.base_crop_jitter = float(self.augmentation_profile.get("bbox_crop_jitter", 0.0))
        self.rear_extra_crop_jitter = float(self.augmentation_profile.get("rear_bbox_crop_jitter", 0.0))
        self.xshape_extra_crop_jitter = float(self.augmentation_profile.get("xshape_bbox_crop_jitter", 0.0))

        base_transforms: list[Any] = [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
        ]
        if augment:
            base_transforms.extend(
                [
                    transforms.ColorJitter(
                        brightness=float(self.augmentation_profile.get("brightness", 0.12)),
                        contrast=float(self.augmentation_profile.get("contrast", 0.12)),
                        saturation=float(self.augmentation_profile.get("saturation", 0.08)),
                    ),
                    transforms.RandomApply(
                        [
                            transforms.RandomAffine(
                                degrees=float(self.augmentation_profile.get("degrees", 4.0)),
                                translate=(
                                    float(self.augmentation_profile.get("translate", 0.04)),
                                    float(self.augmentation_profile.get("translate", 0.04)),
                                ),
                                scale=(
                                    float(self.augmentation_profile.get("scale_min", 0.96)),
                                    float(self.augmentation_profile.get("scale_max", 1.04)),
                                ),
                            )
                        ],
                        p=0.5,
                    ),
                ]
            )
        base_transforms.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
        self.transform = transforms.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def _read_video_frame(self, video_path: str, frame_id: int) -> np.ndarray:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            target_frame = int(np.clip(frame_id, 0, total_frames - 1))
        else:
            target_frame = max(int(frame_id), 0)

        # Some videos are shorter than annotated frame ids or seek imprecisely.
        # Try the requested frame first, then fall back to several nearby earlier frames.
        candidate_frames = [target_frame]
        for delta in (1, 2, 3, 5, 8, 13, 21):
            fallback_frame = max(target_frame - delta, 0)
            if fallback_frame not in candidate_frames:
                candidate_frames.append(fallback_frame)

        frame = None
        for candidate in candidate_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
            ok, candidate_frame = capture.read()
            if ok and candidate_frame is not None:
                frame = candidate_frame
                break

        capture.release()

        if frame is None:
            raise RuntimeError(
                f"Failed to read frame {frame_id} from {video_path} "
                f"(clamped target={target_frame}, total_frames={total_frames})"
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _read_frame(self, pig_id: str, video_path: str, camera: str, frame_id: int) -> np.ndarray:
        if self.frame_cache_dir is not None:
            cache_path = build_cache_frame_path(self.frame_cache_dir, pig_id, camera, frame_id)
            if cache_path.exists():
                frame = cv2.imread(str(cache_path))
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._read_video_frame(video_path, frame_id)

    def _crop_frame(
        self,
        frame: np.ndarray,
        bboxes: list[dict[str, Any]],
        camera: str,
        row: dict[str, Any],
        index: int,
    ) -> np.ndarray:
        if not self.use_bbox_crops or not bboxes:
            return frame

        height, width = frame.shape[:2]
        xyxy_boxes = [bbox_yolo_to_xyxy(item["bbox"], width, height) for item in bboxes]

        x1 = min(box[0] for box in xyxy_boxes)
        y1 = min(box[1] for box in xyxy_boxes)
        x2 = max(box[2] for box in xyxy_boxes)
        y2 = max(box[3] for box in xyxy_boxes)

        pad_ratio = 0.08
        if self.augment:
            pad_ratio += self.base_crop_jitter
            if camera == "rear":
                pad_ratio += self.rear_extra_crop_jitter
            if int(row.get("x_shape", 0)) == 1 and camera == "rear":
                pad_ratio += self.xshape_extra_crop_jitter

        pad_x = max(int(pad_ratio * (x2 - x1)), 4)
        pad_y = max(int(pad_ratio * (y2 - y1)), 4)

        if self.augment and pad_ratio > 0.08:
            rng = random.Random(self.seed + index + int(sum(box[0] for box in xyxy_boxes)))
            jitter_x = int(rng.uniform(-pad_x * 0.35, pad_x * 0.35))
            jitter_y = int(rng.uniform(-pad_y * 0.35, pad_y * 0.35))
        else:
            jitter_x = 0
            jitter_y = 0

        x1 = max(x1 - pad_x, 0)
        y1 = max(y1 - pad_y, 0)
        x2 = min(x2 + pad_x, width)
        y2 = min(y2 + pad_y, height)
        x1 = max(x1 + jitter_x, 0)
        x2 = min(x2 + jitter_x, width)
        y1 = max(y1 + jitter_y, 0)
        y2 = min(y2 + jitter_y, height)

        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]

    def _empty_image(self) -> torch.Tensor:
        return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

    def _sample_camera_frames(
        self,
        pig_id: str,
        video_path: str,
        sampled_frames: list[FrameSample],
        camera: str,
        row: dict[str, Any],
        index: int,
    ) -> tuple[list[torch.Tensor], list[int]]:
        camera_items = [item for item in sampled_frames if item.camera == camera]
        images: list[torch.Tensor] = []
        mask: list[int] = []

        for item in camera_items[: self.frames_per_camera]:
            frame = self._read_frame(pig_id, video_path, camera, item.frame_id)
            frame = self._crop_frame(frame, item.bboxes, camera=camera, row=row, index=index)
            tensor = self.transform(frame)
            images.append(tensor)
            mask.append(1)

        while len(images) < self.frames_per_camera:
            images.append(self._empty_image())
            mask.append(0)

        return images, mask

    def _build_frame_plan(self, row: dict[str, Any], index: int) -> list[FrameSample]:
        annotation_path = str(row.get("annotation_path", "")).strip()
        use_raw_sampling = bool(self.raw_sample_ratio > 0.0 and random.Random(self.seed + index).random() < self.raw_sample_ratio)
        if annotation_path and not use_raw_sampling:
            annotation = load_annotation(annotation_path)
            return collect_frame_samples(
                annotation=annotation,
                frames_per_camera=self.frames_per_camera,
                seed=self.seed + index,
            )

        videos = {camera: row[f"{camera}_video"] for camera in CAMERAS}
        sampled_frames: list[FrameSample] = []
        for camera in CAMERAS:
            sampled_frames.extend(
                collect_uniform_frame_samples(
                    video_path=videos[camera],
                    camera=camera,
                    frames_per_camera=self.frames_per_camera,
                )
            )
        return sampled_frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sampled_frames = self._build_frame_plan(row, index)
        videos = {camera: row[f"{camera}_video"] for camera in CAMERAS}

        all_images: list[torch.Tensor] = []
        all_mask: list[int] = []
        for camera in CAMERAS:
            camera_images, camera_mask = self._sample_camera_frames(
                row["pig_id"],
                videos[camera],
                sampled_frames,
                camera,
                row=row,
                index=index,
            )
            all_images.extend(camera_images)
            all_mask.extend(camera_mask)

        return {
            "pig_id": row["pig_id"],
            "images": torch.stack(all_images, dim=0),
            "frame_mask": torch.tensor(all_mask, dtype=torch.float32),
            "target": torch.tensor(get_target_vector(row), dtype=torch.float32),
            "has_target": torch.tensor(int(row.get("has_target", 1)), dtype=torch.int64),
        }
