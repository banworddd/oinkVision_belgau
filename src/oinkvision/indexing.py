"""Utilities for building animal-level records from annotation JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .constants import CAMERAS, LABELS


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
    num_annotated_frames: int
    annotated_top_frames: int
    annotated_right_frames: int
    annotated_left_frames: int
    annotated_rear_frames: int
    annotation_path: str


def load_annotation(annotation_path: str | Path) -> dict[str, Any]:
    path = Path(annotation_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_camera_annotations(annotations: list[dict[str, Any]], camera: str) -> int:
    return sum(1 for frame in annotations if frame.get(camera))


def build_record(annotation_path: str | Path, data_root: str | Path) -> PigRecord:
    annotation_path = Path(annotation_path)
    data_root = Path(data_root)
    data = load_annotation(annotation_path)

    video = data["video"]
    target = data["target"]
    annotations = data.get("annotations", [])

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
        num_annotated_frames=len(annotations),
        annotated_top_frames=_count_camera_annotations(annotations, "top"),
        annotated_right_frames=_count_camera_annotations(annotations, "right"),
        annotated_left_frames=_count_camera_annotations(annotations, "left"),
        annotated_rear_frames=_count_camera_annotations(annotations, "rear"),
        annotation_path=str(annotation_path),
    )


def validate_record(record: PigRecord) -> list[str]:
    errors: list[str] = []

    for field_name in LABELS:
        value = getattr(record, field_name)
        if value not in (0, 1):
            errors.append(f"{record.pig_id}: label {field_name} must be 0 or 1, got {value}")

    for camera in CAMERAS:
        video_path = Path(getattr(record, f"{camera}_video"))
        if not video_path.exists():
            errors.append(f"{record.pig_id}: missing video for {camera}: {video_path}")

    if record.num_annotated_frames <= 0:
        errors.append(f"{record.pig_id}: no annotated frames found")

    return errors


def build_records(annotation_dir: str | Path, data_root: str | Path) -> list[PigRecord]:
    annotation_dir = Path(annotation_dir)
    paths = sorted(annotation_dir.glob("pig_*.json"))
    records = [build_record(path, data_root) for path in paths]
    return records


def records_to_rows(records: list[PigRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
