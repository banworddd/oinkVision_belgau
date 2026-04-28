"""Tune thresholds for an ensemble in raw-no-metadata replay mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.constants import LABELS, get_active_labels
from oinkvision.dataset import FRONT_META_FIELDS, PigVideoDataset, load_index
from oinkvision.env import get_output_root, load_local_env
from oinkvision.infer import (
    load_config,
    maybe_apply_geometry_fusion,
    maybe_apply_specialist_fusion,
    predict,
)
from oinkvision.metrics import compute_macro_f1
from oinkvision.model import build_model
from oinkvision.train import build_aggregation_spec, choose_device


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, default=output_root / "index" / "train_index.csv")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, default=output_root / "tuned_thresholds_ensemble_raw_replay.json")
    parser.add_argument("--grid-step", type=float, default=0.05)
    return parser.parse_args()


def prepare_raw_nometa_rows(rows: list[dict]) -> list[dict]:
    prepared: list[dict] = []
    for row in rows:
        item = dict(row)
        item["annotation_path"] = ""
        for field in FRONT_META_FIELDS:
            item[field] = 0
        prepared.append(item)
    return prepared


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    active_labels = get_active_labels(config)
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)
    threshold_grid = np.arange(float(args.grid_step), 1.0, float(args.grid_step), dtype=np.float32)

    rows = prepare_raw_nometa_rows(load_index(args.index_path))
    dataset = PigVideoDataset(
        index_path=args.index_path,
        rows=rows,
        frames_per_camera=config["data"]["frames_per_camera"],
        image_size=config["data"]["image_size"],
        use_bbox_crops=config["data"]["use_bbox_crops"],
        frame_cache_dir=config["data"].get("frame_cache_dir"),
        augmentation_profile=config.get("augmentation", {}),
        seed=int(config["seed"]),
        augment=False,
        raw_sample_ratio=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )

    probs_per_checkpoint: list[np.ndarray] = []
    targets_reference: np.ndarray | None = None
    has_target_reference: np.ndarray | None = None

    for checkpoint in args.checkpoints:
        model = build_model(config).to(device)
        state_dict = torch.load(checkpoint, map_location=device)
        load_result = model.load_state_dict(state_dict, strict=False)
        print(
            {
                "checkpoint": str(checkpoint),
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }
        )

        _, targets, main_probs, has_target, aux_probs = predict(
            model,
            loader,
            device,
            aggregation_spec=aggregation_spec,
        )
        probs, specialist_skip_labels = maybe_apply_specialist_fusion(
            rows=rows,
            main_probs=main_probs,
            aux_probs=aux_probs,
            config=config,
            active_labels=active_labels,
        )
        probs = maybe_apply_geometry_fusion(
            rows=rows,
            probs=probs,
            config=config,
            active_labels=active_labels,
            skip_labels=specialist_skip_labels if specialist_skip_labels else None,
        )
        probs_per_checkpoint.append(probs)

        if targets_reference is None:
            targets_reference = targets
            has_target_reference = has_target

    if targets_reference is None or has_target_reference is None:
        raise ValueError("No predictions produced.")
    if not bool(has_target_reference.all()):
        raise ValueError("Raw replay threshold tuning requires labeled rows.")

    probs = np.mean(np.stack(probs_per_checkpoint, axis=0), axis=0)
    target_indices = [LABELS.index(label) for label in active_labels]
    targets_reference = targets_reference[:, target_indices]

    best_thresholds: list[float] = []
    search_scores: dict[str, float] = {}
    for class_idx, label in enumerate(active_labels):
        best_threshold = 0.5
        best_score = -1.0
        class_targets = targets_reference[:, class_idx]
        class_probs = probs[:, class_idx]
        for threshold in threshold_grid:
            class_preds = (class_probs >= float(threshold)).astype(np.int64)
            score = float(f1_score(class_targets, class_preds, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        best_thresholds.append(best_threshold)
        search_scores[label] = best_score

    tuned_metrics = compute_macro_f1(targets_reference, probs, thresholds=best_thresholds, labels=active_labels)
    payload = {
        "mode": "raw_nometa_ensemble",
        "index_path": str(args.index_path),
        "checkpoints": [str(p) for p in args.checkpoints],
        "thresholds": best_thresholds,
        "threshold_search_scores": search_scores,
        "tuned_metrics": tuned_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Ensemble raw replay thresholds saved to: {args.output_json}")
    print(payload)


if __name__ == "__main__":
    main()
