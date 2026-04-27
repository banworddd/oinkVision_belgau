"""Geometry helpers for hind-limb heuristics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _bbox_center_width_height(bbox: list[float]) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = bbox
    return float(x_center), float(y_center), float(width), float(height)


def rear_leg_geometry_features(rear_bboxes: list[dict[str, Any]]) -> dict[str, float]:
    left_box = None
    right_box = None
    for item in rear_bboxes:
        label = item.get("label", "")
        if label == "left_back_leg":
            left_box = item["bbox"]
        elif label == "right_back_leg":
            right_box = item["bbox"]

    if left_box is None or right_box is None:
        return {
            "has_both_legs": 0.0,
            "center_dx": 0.0,
            "center_dy": 0.0,
            "avg_width": 0.0,
            "avg_height": 0.0,
            "width_ratio": 0.0,
            "height_ratio": 0.0,
            "normalized_gap": 0.0,
            "xshape_score": 0.0,
        }

    lx, ly, lw, lh = _bbox_center_width_height(left_box)
    rx, ry, rw, rh = _bbox_center_width_height(right_box)

    center_dx = abs(rx - lx)
    center_dy = abs(ry - ly)
    avg_width = (lw + rw) / 2.0
    avg_height = (lh + rh) / 2.0
    width_ratio = min(lw, rw) / max(lw, rw, 1e-6)
    height_ratio = min(lh, rh) / max(lh, rh, 1e-6)
    normalized_gap = center_dx / max(avg_width, 1e-6)

    # Higher score when legs are unusually close in x and roughly aligned in size.
    proximity_score = max(0.0, 1.5 - normalized_gap) / 1.5
    symmetry_score = 0.5 * (width_ratio + height_ratio)
    xshape_score = float(np.clip(0.65 * proximity_score + 0.35 * symmetry_score, 0.0, 1.0))

    return {
        "has_both_legs": 1.0,
        "center_dx": float(center_dx),
        "center_dy": float(center_dy),
        "avg_width": float(avg_width),
        "avg_height": float(avg_height),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
        "normalized_gap": float(normalized_gap),
        "xshape_score": xshape_score,
    }


def side_leg_geometry_features(side_bboxes: list[dict[str, Any]]) -> dict[str, float]:
    if not side_bboxes:
        return {
            "has_leg": 0.0,
            "aspect_ratio": 0.0,
            "soft_pastern_score": 0.0,
        }

    bbox = side_bboxes[0]["bbox"]
    _, _, width, height = _bbox_center_width_height(bbox)
    aspect_ratio = width / max(height, 1e-6)

    # Softer pastern is approximated by a relatively wider / less elongated side-leg box.
    soft_pastern_score = float(np.clip((aspect_ratio - 0.18) / 0.18, 0.0, 1.0))
    return {
        "has_leg": 1.0,
        "aspect_ratio": float(aspect_ratio),
        "soft_pastern_score": soft_pastern_score,
    }


def aggregate_rear_geometry(annotation: dict[str, Any]) -> dict[str, float]:
    per_frame = []
    for frame in annotation.get("annotations", []):
        rear_items = frame.get("rear", [])
        if not rear_items:
            continue
        features = rear_leg_geometry_features(rear_items)
        if features["has_both_legs"] > 0:
            per_frame.append(features)

    if not per_frame:
        return {"num_rear_frames": 0.0, "mean_xshape_score": 0.0, "max_xshape_score": 0.0}

    scores = [item["xshape_score"] for item in per_frame]
    return {
        "num_rear_frames": float(len(per_frame)),
        "mean_xshape_score": float(np.mean(scores)),
        "max_xshape_score": float(np.max(scores)),
    }


def aggregate_side_geometry(annotation: dict[str, Any], camera: str) -> dict[str, float]:
    per_frame = []
    for frame in annotation.get("annotations", []):
        side_items = frame.get(camera, [])
        if not side_items:
            continue
        features = side_leg_geometry_features(side_items)
        if features["has_leg"] > 0:
            per_frame.append(features)

    prefix = camera
    if not per_frame:
        return {
            f"num_{prefix}_frames": 0.0,
            f"mean_{prefix}_soft_pastern_score": 0.0,
            f"max_{prefix}_soft_pastern_score": 0.0,
        }

    scores = [item["soft_pastern_score"] for item in per_frame]
    return {
        f"num_{prefix}_frames": float(len(per_frame)),
        f"mean_{prefix}_soft_pastern_score": float(np.mean(scores)),
        f"max_{prefix}_soft_pastern_score": float(np.max(scores)),
    }


def aggregate_annotation_geometry(annotation: dict[str, Any]) -> dict[str, float]:
    rear = aggregate_rear_geometry(annotation)
    left = aggregate_side_geometry(annotation, camera="left")
    right = aggregate_side_geometry(annotation, camera="right")

    max_side_soft = max(left["max_left_soft_pastern_score"], right["max_right_soft_pastern_score"])
    mean_side_soft = 0.5 * (left["mean_left_soft_pastern_score"] + right["mean_right_soft_pastern_score"])

    # A weak posture helper: unusual rear geometry and soft-pastern-like side geometry together.
    max_bad_posture_score = float(np.clip(0.6 * rear["max_xshape_score"] + 0.4 * max_side_soft, 0.0, 1.0))
    mean_bad_posture_score = float(np.clip(0.6 * rear["mean_xshape_score"] + 0.4 * mean_side_soft, 0.0, 1.0))

    return {
        **rear,
        **left,
        **right,
        "max_soft_pastern_score": float(max_side_soft),
        "mean_soft_pastern_score": float(mean_side_soft),
        "max_bad_posture_score": max_bad_posture_score,
        "mean_bad_posture_score": mean_bad_posture_score,
    }
