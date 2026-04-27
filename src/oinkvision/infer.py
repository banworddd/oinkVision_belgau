"""Inference utilities for validation and submission generation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oinkvision.constants import LABELS
from oinkvision.dataset import PigVideoDataset, load_annotation, load_index
from oinkvision.env import apply_env_overrides, get_output_root, load_local_env
from oinkvision.geometry import aggregate_annotation_geometry
from oinkvision.metrics import apply_thresholds, compute_macro_f1
from oinkvision.model import build_model
from oinkvision.train import aggregate_frame_logits, build_aggregation_spec, choose_device


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=output_root / "index" / "train_index.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=output_root / "best_model.pt",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=output_root / "validation_predictions.csv",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=output_root / "validation_metrics.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--submission-only",
        action="store_true",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return apply_env_overrides(config)


def build_loader(config: dict[str, Any], index_path: Path, limit: int | None) -> DataLoader:
    rows = load_index(index_path)
    if limit is not None:
        rows = rows[:limit]

    dataset = PigVideoDataset(
        index_path=index_path,
        rows=rows,
        frames_per_camera=config["data"]["frames_per_camera"],
        image_size=config["data"]["image_size"],
        use_bbox_crops=config["data"]["use_bbox_crops"],
        frame_cache_dir=config["data"].get("frame_cache_dir"),
        augmentation_profile=config.get("augmentation", {}),
        seed=config["seed"],
    )
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )


def load_rows_for_index(index_path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = load_index(index_path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def maybe_apply_geometry_fusion(
    rows: list[dict[str, Any]],
    probs: np.ndarray,
    config: dict[str, Any],
    postprocess_params: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = dict(config.get("postprocess", {}))
    if postprocess_params is not None:
        cfg.update(postprocess_params)

    if not bool(cfg.get("enable_geometry_fusion", False)):
        return probs

    per_class_cfg = cfg.get("geometry_fusion", {})
    fused = probs.copy()
    for row_idx, row in enumerate(rows):
        annotation_path = str(row.get("annotation_path", "")).strip()
        if not annotation_path:
            continue
        features = aggregate_annotation_geometry(load_annotation(annotation_path))
        for label_idx, label in enumerate(LABELS):
            label_cfg = per_class_cfg.get(label)
            if not label_cfg:
                continue
            feature_name = str(label_cfg.get("feature", "")).strip()
            if not feature_name:
                continue
            alpha = float(label_cfg.get("alpha", 0.0))
            mode = str(label_cfg.get("mode", "max_scaled"))
            feature_value = float(features.get(feature_name, 0.0))
            if mode == "weighted_sum":
                fused[row_idx, label_idx] = (1.0 - alpha) * fused[row_idx, label_idx] + alpha * feature_value
            else:
                fused[row_idx, label_idx] = max(fused[row_idx, label_idx], alpha * feature_value)
    return fused


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    aggregation_spec: dict[str, Any] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pig_ids: list[str] = []
    all_targets = []
    all_probs = []
    all_has_target = []

    for batch in tqdm(loader, leave=False):
        images = batch["images"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        targets = batch["target"].cpu().numpy()
        has_target = batch["has_target"].cpu().numpy()

        batch_size, num_frames, channels, height, width = images.shape
        flat_images = images.view(batch_size * num_frames, channels, height, width)
        frame_logits = model(flat_images).view(batch_size, num_frames, len(LABELS))
        logits = aggregate_frame_logits(frame_logits, frame_mask, aggregation_spec=aggregation_spec)
        probs = torch.sigmoid(logits).cpu().numpy()

        pig_ids.extend(batch["pig_id"])
        all_targets.append(targets)
        all_probs.append(probs)
        all_has_target.append(has_target)

    targets_np = np.concatenate(all_targets, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)
    has_target_np = np.concatenate(all_has_target, axis=0)
    return pig_ids, targets_np, probs_np, has_target_np


def write_predictions(
    pig_ids: list[str],
    probs: np.ndarray,
    preds: np.ndarray,
    output_csv: Path,
    submission_only: bool = False,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if submission_only:
            writer.writerow(["id", "bad_posture", "bumps", "soft_pastern", "x_shape"])
        else:
            writer.writerow(
                [
                    "id",
                    "bad_posture_prob",
                    "bumps_prob",
                    "soft_pastern_prob",
                    "x_shape_prob",
                    "bad_posture",
                    "bumps",
                    "soft_pastern",
                    "x_shape",
                ]
            )
        for pig_id, prob_row, pred_row in zip(pig_ids, probs, preds):
            if submission_only:
                writer.writerow([pig_id, *[int(x) for x in pred_row]])
            else:
                writer.writerow(
                    [
                        pig_id,
                        *[float(x) for x in prob_row],
                        *[int(x) for x in pred_row],
                    ]
                )


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)

    model = build_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    rows = load_rows_for_index(args.index_path, args.limit)
    loader = build_loader(config, args.index_path, args.limit)
    pig_ids, targets, probs, has_target = predict(model, loader, device, aggregation_spec=aggregation_spec)

    postprocess_params = None
    if args.thresholds_json is not None:
        with args.thresholds_json.open("r", encoding="utf-8") as f:
            threshold_data = json.load(f)
        thresholds = [float(x) for x in threshold_data["thresholds"]]
        postprocess_params = threshold_data.get("postprocess")
    else:
        thresholds = [float(config["inference"]["thresholds"][label]) for label in LABELS]

    probs = maybe_apply_geometry_fusion(rows, probs, config, postprocess_params=postprocess_params)
    preds = apply_thresholds(probs, thresholds)
    metrics = None
    if bool(has_target.all()):
        metrics = compute_macro_f1(targets, probs, thresholds=thresholds)

    write_predictions(pig_ids, probs, preds, args.output_csv, submission_only=bool(args.submission_only))
    if metrics is not None:
        write_metrics(metrics, args.metrics_json)

    print(f"Predictions saved to: {args.output_csv}")
    if metrics is not None:
        print(f"Metrics saved to: {args.metrics_json}")
        print(metrics)
    else:
        print("Targets are unavailable for this index, so metrics were not computed.")


if __name__ == "__main__":
    main()
