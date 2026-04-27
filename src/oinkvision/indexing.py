"""Utilities for building animal-level records from annotation JSON files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

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
class PigRecord:
    pig_id: str
    top_video: str
    right_video: str
    left_video: str
    rear_video: str
    bad_posture: int
    bumps: int
    soft_pastern: int
    x_shape: int
    has_target: int
    num_annotated_frames: int
    annotated_top_frames: int
    annotated_right_frames: int
    annotated_left_frames: int
    annotated_rear_frames: int
    annotation_path: str
    front_left_bad_posture: int
    front_right_bad_posture: int
    front_left_bumps: int
    front_right_bumps: int
    front_left_soft_pastern: int
    front_right_soft_pastern: int
    front_left_x_shape: int
    front_right_x_shape: int


def load_annotation(annotation_path: str | Path) -> dict[str, Any]:
    path = Path(annotation_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_camera_annotations(annotations: list[dict[str, Any]], camera: str) -> int:
    return sum(1 for frame in annotations if frame.get(camera))


VIDEO_NAME_RE = re.compile(r"^pig_(?P<pig_id>.+?)_cam_(?P<camera>top|right|left|rear)\.mp4$")


def _normalize_pig_key(value: str | int) -> str:
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return str(int(digits)) if digits else text.lower()


def load_front_metadata(metadata_path: str | Path | None) -> dict[str, dict[str, int]]:
    if metadata_path is None:
        return {}
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}

    table = pd.read_excel(metadata_path)
    required_columns = ["ID", *FRONT_META_FIELDS]
    missing = [name for name in required_columns if name not in table.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns in {metadata_path}: {missing}")

    metadata_by_pig: dict[str, dict[str, int]] = {}
    for _, row in table.iterrows():
        pig_key = _normalize_pig_key(row["ID"])
        metadata_by_pig[pig_key] = {field: int(bool(row[field])) for field in FRONT_META_FIELDS}
    return metadata_by_pig


def build_record(
    annotation_path: str | Path,
    data_root: str | Path,
    metadata_by_pig: dict[str, dict[str, int]] | None = None,
) -> PigRecord:
    annotation_path = Path(annotation_path)
    data_root = Path(data_root)
    data = load_annotation(annotation_path)

    video = data["video"]
    target = data["target"]
    annotations = data.get("annotations", [])

    metadata_by_pig = metadata_by_pig or {}
    front_meta = metadata_by_pig.get(_normalize_pig_key(data["pig_id"]), {})

    return PigRecord(
        pig_id=str(data["pig_id"]),
        top_video=str(data_root / video["top"]),
        right_video=str(data_root / video["right"]),
        left_video=str(data_root / video["left"]),
        rear_video=str(data_root / video["rear"]),
        bad_posture=int(target["bad_posture"]),
        bumps=int(target["bumps"]),
        soft_pastern=int(target["soft_pastern"]),
        x_shape=int(target["x_shape"]),
        has_target=1,
        num_annotated_frames=len(annotations),
        annotated_top_frames=_count_camera_annotations(annotations, "top"),
        annotated_right_frames=_count_camera_annotations(annotations, "right"),
        annotated_left_frames=_count_camera_annotations(annotations, "left"),
        annotated_rear_frames=_count_camera_annotations(annotations, "rear"),
        annotation_path=str(annotation_path),
        front_left_bad_posture=int(front_meta.get("front_left_bad_posture", 0)),
        front_right_bad_posture=int(front_meta.get("front_right_bad_posture", 0)),
        front_left_bumps=int(front_meta.get("front_left_bumps", 0)),
        front_right_bumps=int(front_meta.get("front_right_bumps", 0)),
        front_left_soft_pastern=int(front_meta.get("front_left_soft_pastern", 0)),
        front_right_soft_pastern=int(front_meta.get("front_right_soft_pastern", 0)),
        front_left_x_shape=int(front_meta.get("front_left_x_shape", 0)),
        front_right_x_shape=int(front_meta.get("front_right_x_shape", 0)),
    )


def validate_record(record: PigRecord) -> list[str]:
    errors: list[str] = []

    if record.has_target:
        for field_name in LABELS:
            value = getattr(record, field_name)
            if value not in (0, 1):
                errors.append(f"{record.pig_id}: label {field_name} must be 0 or 1, got {value}")

    for camera in CAMERAS:
        video_path = Path(getattr(record, f"{camera}_video"))
        if not video_path.exists():
            errors.append(f"{record.pig_id}: missing video for {camera}: {video_path}")

    if record.has_target and record.num_annotated_frames <= 0:
        errors.append(f"{record.pig_id}: no annotated frames found")

    return errors


def build_records(
    annotation_dir: str | Path,
    data_root: str | Path,
    metadata_path: str | Path | None = None,
) -> list[PigRecord]:
    annotation_dir = Path(annotation_dir)
    paths = sorted(annotation_dir.glob("pig_*.json"))
    metadata_by_pig = load_front_metadata(metadata_path)
    records = [build_record(path, data_root, metadata_by_pig=metadata_by_pig) for path in paths]
    return records


def build_record_from_raw_group(
    pig_id: str,
    grouped_paths: dict[str, Path],
    metadata_by_pig: dict[str, dict[str, int]] | None = None,
) -> PigRecord:
    missing = [camera for camera in CAMERAS if camera not in grouped_paths]
    if missing:
        raise ValueError(f"{pig_id}: missing camera videos: {missing}")

    metadata_by_pig = metadata_by_pig or {}
    front_meta = metadata_by_pig.get(_normalize_pig_key(pig_id), {})
    return PigRecord(
        pig_id=str(pig_id),
        top_video=str(grouped_paths["top"]),
        right_video=str(grouped_paths["right"]),
        left_video=str(grouped_paths["left"]),
        rear_video=str(grouped_paths["rear"]),
        bad_posture=0,
        bumps=0,
        soft_pastern=0,
        x_shape=0,
        has_target=0,
        num_annotated_frames=0,
        annotated_top_frames=0,
        annotated_right_frames=0,
        annotated_left_frames=0,
        annotated_rear_frames=0,
        annotation_path="",
        front_left_bad_posture=int(front_meta.get("front_left_bad_posture", 0)),
        front_right_bad_posture=int(front_meta.get("front_right_bad_posture", 0)),
        front_left_bumps=int(front_meta.get("front_left_bumps", 0)),
        front_right_bumps=int(front_meta.get("front_right_bumps", 0)),
        front_left_soft_pastern=int(front_meta.get("front_left_soft_pastern", 0)),
        front_right_soft_pastern=int(front_meta.get("front_right_soft_pastern", 0)),
        front_left_x_shape=int(front_meta.get("front_left_x_shape", 0)),
        front_right_x_shape=int(front_meta.get("front_right_x_shape", 0)),
    )


def build_records_from_raw_dir(raw_dir: str | Path, metadata_path: str | Path | None = None) -> list[PigRecord]:
    raw_dir = Path(raw_dir)
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(raw_dir.glob("pig_*_cam_*.mp4")):
        match = VIDEO_NAME_RE.match(path.name)
        if match is None:
            continue
        pig_id = match.group("pig_id")
        camera = match.group("camera")
        grouped.setdefault(pig_id, {})[camera] = path

    metadata_by_pig = load_front_metadata(metadata_path)
    records = [
        build_record_from_raw_group(pig_id, grouped_paths, metadata_by_pig=metadata_by_pig)
        for pig_id, grouped_paths in sorted(grouped.items())
    ]
    return records


def records_to_rows(records: list[PigRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
