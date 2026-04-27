"""Tune class thresholds in raw-no-metadata replay mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.constants import LABELS
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=output_root / "tuned_thresholds_raw_replay.json")
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
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)
    grid_step = float(args.grid_step)
    threshold_grid = np.arange(grid_step, 1.0, grid_step, dtype=np.float32)

    rows = load_index(args.index_path)
    rows = prepare_raw_nometa_rows(rows)
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

    model = build_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    _, targets, main_probs, has_target, aux_probs = predict(
        model,
        loader,
        device,
        aggregation_spec=aggregation_spec,
    )
    if not bool(has_target.all()):
        raise ValueError("Raw replay threshold tuning requires labeled rows.")

    probs, specialist_skip_labels = maybe_apply_specialist_fusion(
        rows=rows,
        main_probs=main_probs,
        aux_probs=aux_probs,
        config=config,
    )
    probs = maybe_apply_geometry_fusion(
        rows=rows,
        probs=probs,
        config=config,
        skip_labels=specialist_skip_labels if specialist_skip_labels else None,
    )

    best_thresholds: list[float] = []
    search_scores: dict[str, float] = {}
    for class_idx, label in enumerate(LABELS):
        best_threshold = 0.5
        best_score = -1.0
        class_targets = targets[:, class_idx]
        class_probs = probs[:, class_idx]
        for threshold in threshold_grid:
            metrics = compute_macro_f1(
                targets[:, class_idx : class_idx + 1],
                class_probs[:, None],
                thresholds=[float(threshold)],
            )
            score = float(next(iter(metrics["per_class_f1"].values())))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        best_thresholds.append(best_threshold)
        search_scores[label] = best_score

    tuned_metrics = compute_macro_f1(targets, probs, thresholds=best_thresholds)
    payload = {
        "mode": "raw_nometa",
        "index_path": str(args.index_path),
        "checkpoint": str(args.checkpoint),
        "thresholds": best_thresholds,
        "threshold_search_scores": search_scores,
        "tuned_metrics": tuned_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Raw replay thresholds saved to: {args.output_json}")
    print(payload)


if __name__ == "__main__":
    main()
