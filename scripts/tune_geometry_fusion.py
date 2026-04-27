"""Tune geometry-based post-processing on a labeled validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.constants import LABELS
from oinkvision.geometry import aggregate_annotation_geometry
from oinkvision.infer import build_loader, load_config, load_rows_for_index, maybe_apply_xshape_specialist_fusion, predict
from oinkvision.metrics import compute_macro_f1
from oinkvision.model import build_model
from oinkvision.train import build_aggregation_spec, choose_device
from oinkvision.dataset import load_annotation
from oinkvision.env import get_output_root, load_local_env


DEFAULT_ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=output_root / "geometry_fusion_tuned.json")
    return parser.parse_args()


def build_geometry_table(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    geometry_rows = []
    for row in rows:
        annotation_path = str(row.get("annotation_path", "")).strip()
        if not annotation_path:
            geometry_rows.append({})
            continue
        geometry_rows.append(aggregate_annotation_geometry(load_annotation(annotation_path)))
    return geometry_rows


def apply_single_class_fusion(
    base_probs: np.ndarray,
    geometry_rows: list[dict[str, float]],
    class_index: int,
    feature_name: str,
    alpha: float,
    mode: str,
) -> np.ndarray:
    fused = base_probs.copy()
    for row_idx, features in enumerate(geometry_rows):
        if not features:
            continue
        feature_value = float(features.get(feature_name, 0.0))
        if mode == "weighted_sum":
            fused[row_idx, class_index] = (1.0 - alpha) * fused[row_idx, class_index] + alpha * feature_value
        else:
            fused[row_idx, class_index] = max(fused[row_idx, class_index], alpha * feature_value)
    return fused


def best_threshold_for_class(targets: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in DEFAULT_THRESHOLD_GRID:
        preds = (probs >= threshold).astype(np.int64)
        score = f1_score(targets, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)

    model = build_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    rows = load_rows_for_index(args.index_path, limit=None)
    loader = build_loader(config, args.index_path, limit=None)
    _, targets, main_probs, has_target, aux_probs = predict(
        model,
        loader,
        device,
        aggregation_spec=aggregation_spec,
    )
    if not bool(has_target.all()):
        raise ValueError("Geometry fusion tuning requires labeled rows with annotation paths.")
    probs, _ = maybe_apply_xshape_specialist_fusion(
        rows=rows,
        main_probs=main_probs,
        aux_probs=aux_probs,
        config=config,
    )

    geometry_rows = build_geometry_table(rows)
    post_cfg = config.get("postprocess", {})
    fusion_cfg = post_cfg.get("geometry_fusion", {})

    tuned_fusion = {}
    fused_probs = probs.copy()
    thresholds = [0.5] * len(LABELS)
    search_scores = {}

    for class_index, label in enumerate(LABELS):
        class_cfg = fusion_cfg.get(label)
        if not class_cfg:
            threshold, score = best_threshold_for_class(targets[:, class_index], probs[:, class_index])
            thresholds[class_index] = threshold
            search_scores[label] = score
            continue

        feature_name = str(class_cfg.get("feature", "")).strip()
        mode = str(class_cfg.get("mode", "max_scaled"))
        if not feature_name:
            threshold, score = best_threshold_for_class(targets[:, class_index], probs[:, class_index])
            thresholds[class_index] = threshold
            search_scores[label] = score
            continue

        best_alpha = 0.0
        best_threshold = 0.5
        best_score = -1.0
        best_probs = probs[:, class_index].copy()
        for alpha in DEFAULT_ALPHA_GRID:
            candidate_probs = apply_single_class_fusion(
                base_probs=probs.copy(),
                geometry_rows=geometry_rows,
                class_index=class_index,
                feature_name=feature_name,
                alpha=float(alpha),
                mode=mode,
            )[:, class_index]
            threshold, score = best_threshold_for_class(targets[:, class_index], candidate_probs)
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_threshold = float(threshold)
                best_probs = candidate_probs.copy()

        fused_probs[:, class_index] = best_probs
        thresholds[class_index] = best_threshold
        search_scores[label] = best_score
        tuned_fusion[label] = {
            "feature": feature_name,
            "mode": mode,
            "alpha": best_alpha,
        }

    tuned_metrics = compute_macro_f1(targets, fused_probs, thresholds=thresholds)
    payload = {
        "thresholds": thresholds,
        "tuned_metrics": tuned_metrics,
        "postprocess": {
            "enable_geometry_fusion": True,
            "geometry_fusion": tuned_fusion,
            "search_scores": search_scores,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Geometry fusion config saved to: {args.output_json}")
    print(payload)


if __name__ == "__main__":
    main()
