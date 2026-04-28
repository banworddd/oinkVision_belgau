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

FRONT_META_FIELDS = [
    "front_left_bad_posture",
    "front_right_bad_posture",
    "front_left_bumps",
    "front_right_bumps",
    "front_left_soft_pastern",
    "front_right_soft_pastern",
    "front_left_x_shape",
    "front_right_x_shape",
]


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
        *FRONT_META_FIELDS,
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


def get_front_meta_vector(row: dict[str, Any]) -> list[float]:
    return [float(int(row.get(field, 0))) for field in FRONT_META_FIELDS]


def collect_frame_samples(
    annotation: dict[str, Any],
    frames_per_camera: int,
    seed: int | None = None,
) -> list[FrameSample]:
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
            selected = select_diverse_annotated_samples(candidates, frames_per_camera)

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
    num_candidates: int | None = None,
) -> list[FrameSample]:
    num_frames = get_video_num_frames(video_path)
    candidate_count = max(frames_per_camera, int(num_candidates or frames_per_camera * 4))
    if candidate_count <= 1:
        frame_ids = [max(num_frames // 2, 0)]
    else:
        frame_ids = np.linspace(0, max(num_frames - 1, 0), num=candidate_count, dtype=int).tolist()
    candidates = [
        FrameSample(
            frame_id=int(frame_id),
            camera=camera,
            bboxes=[],
            source="uniform",
        )
        for frame_id in frame_ids
    ]
    if len(candidates) <= frames_per_camera:
        return candidates
    return select_diverse_raw_samples(video_path=video_path, camera=camera, candidates=candidates, top_k=frames_per_camera)


def _bbox_quality_score(sample: FrameSample) -> float:
    boxes = sample.bboxes
    if not boxes:
        return 0.0
    areas = []
    center_scores = []
    x_centers = []
    heights = []
    widths = []
    for item in boxes:
        x_center, y_center, width, height = [float(x) for x in item["bbox"]]
        areas.append(width * height)
        center_dist = float(np.hypot(x_center - 0.5, y_center - 0.5))
        center_scores.append(max(0.0, 1.0 - center_dist / 0.75))
        x_centers.append(x_center)
        heights.append(height)
        widths.append(width)

    area_score = min(float(sum(areas)) / 0.35, 1.0)
    center_score = float(np.mean(center_scores))
    multi_box_score = min(len(boxes) / 2.0, 1.0)
    score = 0.55 * area_score + 0.25 * center_score + 0.20 * multi_box_score

    if sample.camera == "rear" and len(boxes) >= 2:
        x_centers = np.asarray(x_centers, dtype=np.float32)
        heights = np.asarray(heights, dtype=np.float32)
        widths = np.asarray(widths, dtype=np.float32)
        separation = float(np.clip((x_centers.max() - x_centers.min()) / 0.55, 0.0, 1.0))
        height_similarity = 1.0 - float(np.clip(abs(heights.max() - heights.min()) / max(heights.max(), 1e-6), 0.0, 1.0))
        width_similarity = 1.0 - float(np.clip(abs(widths.max() - widths.min()) / max(widths.max(), 1e-6), 0.0, 1.0))
        symmetry = 0.5 * height_similarity + 0.5 * width_similarity
        score += 0.15 * separation + 0.10 * symmetry

    return float(score)


def _select_diverse_topk(
    candidates: list[FrameSample],
    top_k: int,
    score_fn,
) -> list[FrameSample]:
    ordered = sorted(candidates, key=lambda item: item.frame_id)
    if len(ordered) <= top_k:
        return ordered

    scores = [float(score_fn(item)) for item in ordered]
    bins = np.array_split(np.arange(len(ordered)), top_k)
    selected_indices: list[int] = []
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        best_idx = max(bin_indices.tolist(), key=lambda idx: scores[idx])
        selected_indices.append(int(best_idx))

    selected_set = set(selected_indices)
    if len(selected_indices) < top_k:
        remaining = sorted(
            (idx for idx in range(len(ordered)) if idx not in selected_set),
            key=lambda idx: scores[idx],
            reverse=True,
        )
        for idx in remaining:
            selected_indices.append(int(idx))
            selected_set.add(int(idx))
            if len(selected_indices) >= top_k:
                break

    selected = [ordered[idx] for idx in sorted(selected_indices[:top_k], key=lambda idx: ordered[idx].frame_id)]
    return selected


def select_diverse_annotated_samples(candidates: list[FrameSample], top_k: int) -> list[FrameSample]:
    return _select_diverse_topk(candidates, top_k=top_k, score_fn=_bbox_quality_score)


def _read_video_frame_direct(video_path: str | Path, frame_id: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        target = int(np.clip(frame_id, 0, total_frames - 1))
    else:
        target = max(int(frame_id), 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _raw_frame_quality_score(frame: np.ndarray, camera: str) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    height, width = frame.shape[:2]
    area_ratio = area / max(float(height * width), 1.0)
    x, y, w, h = cv2.boundingRect(contour)
    cx = (x + w / 2.0) / max(width, 1)
    cy = (y + h / 2.0) / max(height, 1)
    center_dist = float(np.hypot(cx - 0.5, cy - 0.5))
    center_score = max(0.0, 1.0 - center_dist / 0.75)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness_score = min(sharpness / 300.0, 1.0)

    score = 0.50 * min(area_ratio / 0.30, 1.0) + 0.25 * center_score + 0.25 * sharpness_score
    if camera == "rear":
        crop_mask = mask[y : y + h, x : x + w]
        if crop_mask.size > 0:
            half = crop_mask.shape[1] // 2
            if half > 0:
                left_mass = float(crop_mask[:, :half].sum())
                right_mass = float(crop_mask[:, half:].sum())
                denom = max(left_mass + right_mass, 1.0)
                symmetry = 1.0 - abs(left_mass - right_mass) / denom
                score += 0.15 * float(np.clip(symmetry, 0.0, 1.0))
    return float(score)


def select_diverse_raw_samples(
    video_path: str | Path,
    camera: str,
    candidates: list[FrameSample],
    top_k: int,
) -> list[FrameSample]:
    cache: dict[int, float] = {}

    def score_fn(sample: FrameSample) -> float:
        if sample.frame_id not in cache:
            frame = _read_video_frame_direct(video_path, sample.frame_id)
            cache[sample.frame_id] = _raw_frame_quality_score(frame, camera)
        return cache[sample.frame_id]

    return _select_diverse_topk(candidates, top_k=top_k, score_fn=score_fn)


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
        detector_cfg = self.augmentation_profile.get("detector_crop", {})
        self.enable_detector_crop_fallback = bool(detector_cfg.get("enabled", False))
        self.detector_min_area_ratio = float(detector_cfg.get("min_area_ratio", 0.02))
        self.detector_expand_ratio = float(detector_cfg.get("expand_ratio", 0.15))
        rear_norm_cfg = self.augmentation_profile.get("rear_view_normalization", {})
        self.enable_rear_view_normalization = bool(rear_norm_cfg.get("enabled", False))
        self.rear_clahe_clip_limit = float(rear_norm_cfg.get("clahe_clip_limit", 2.0))
        self.rear_clahe_tile = int(rear_norm_cfg.get("clahe_tile", 8))
        self.synthetic_xshape_flag_key = str(self.augmentation_profile.get("xshape_synthetic_flag_key", "is_xshape_augmented"))
        self.color_only_for_synthetic_xshape = bool(
            self.augmentation_profile.get("color_only_for_synthetic_xshape", True)
        )
        self.disable_geometry_for_synthetic_xshape = bool(
            self.augmentation_profile.get("disable_geometry_for_synthetic_xshape", True)
        )
        self.affine_prob = float(self.augmentation_profile.get("affine_prob", 0.5))
        self.raw_candidate_multiplier = int(self.augmentation_profile.get("raw_candidate_multiplier", 4))
        self.raw_candidate_cap = int(self.augmentation_profile.get("raw_candidate_cap", 32))

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
            ]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=float(self.augmentation_profile.get("brightness", 0.12)),
            contrast=float(self.augmentation_profile.get("contrast", 0.12)),
            saturation=float(self.augmentation_profile.get("saturation", 0.08)),
        )
        self.affine = transforms.RandomAffine(
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
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _is_synthetic_xshape_row(self, row: dict[str, Any]) -> bool:
        return int(row.get("x_shape", 0)) == 1 and int(row.get(self.synthetic_xshape_flag_key, 0) or 0) == 1

    def _apply_transforms(self, frame: np.ndarray, row: dict[str, Any]) -> torch.Tensor:
        tensor = self.preprocess(frame)
        if not self.augment:
            return self.normalize(tensor)

        tensor = self.color_jitter(tensor)
        apply_geometry = True
        if self.disable_geometry_for_synthetic_xshape and self._is_synthetic_xshape_row(row):
            apply_geometry = False
        if apply_geometry and random.random() < self.affine_prob:
            tensor = self.affine(tensor)

        if self.color_only_for_synthetic_xshape and self._is_synthetic_xshape_row(row):
            return self.normalize(tensor)
        return self.normalize(tensor)

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

    def _detect_foreground_bbox(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        height, width = frame.shape[:2]
        if area < self.detector_min_area_ratio * float(height * width):
            return None
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            return None
        pad_x = int(w * self.detector_expand_ratio)
        pad_y = int(h * self.detector_expand_ratio)
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, width)
        y2 = min(y + h + pad_y, height)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _normalize_rear_view(self, crop: np.ndarray) -> np.ndarray:
        if not self.enable_rear_view_normalization:
            return crop
        if crop.size == 0:
            return crop
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.rear_clahe_clip_limit,
            tileGridSize=(self.rear_clahe_tile, self.rear_clahe_tile),
        )
        l_channel = clahe.apply(l_channel)
        normalized = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)

    def _crop_frame(
        self,
        frame: np.ndarray,
        bboxes: list[dict[str, Any]],
        camera: str,
        row: dict[str, Any],
        index: int,
    ) -> np.ndarray:
        if not self.use_bbox_crops:
            return frame

        height, width = frame.shape[:2]
        if bboxes:
            xyxy_boxes = [bbox_yolo_to_xyxy(item["bbox"], width, height) for item in bboxes]
        elif self.enable_detector_crop_fallback:
            detected = self._detect_foreground_bbox(frame)
            if detected is None:
                return frame
            xyxy_boxes = [detected]
        else:
            return frame

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
        crop = frame[y1:y2, x1:x2]
        if camera == "rear":
            crop = self._normalize_rear_view(crop)
        return crop

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
            tensor = self._apply_transforms(frame, row=row)
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
            candidate_count = min(self.raw_candidate_cap, max(self.frames_per_camera, self.frames_per_camera * self.raw_candidate_multiplier))
            sampled_frames.extend(
                collect_uniform_frame_samples(
                    video_path=videos[camera],
                    camera=camera,
                    frames_per_camera=self.frames_per_camera,
                    num_candidates=candidate_count,
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
            "front_meta": torch.tensor(get_front_meta_vector(row), dtype=torch.float32),
            "has_target": torch.tensor(int(row.get("has_target", 1)), dtype=torch.int64),
        }
