"""Run ensemble inference by averaging probabilities from multiple checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.constants import LABELS
from oinkvision.infer import (
    build_loader,
    load_config,
    load_rows_for_index,
    maybe_apply_geometry_fusion,
    maybe_apply_specialist_fusion,
    predict,
)
from oinkvision.metrics import apply_thresholds, compute_macro_f1
from oinkvision.model import build_model
from oinkvision.train import build_aggregation_spec, choose_device
from oinkvision.env import get_output_root, load_local_env
from oinkvision.xshape_specialist import (
    compute_xshape_specialist_scores,
    extract_ensemble_rear_embeddings,
    prepare_raw_nometa_rows,
)


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=output_root / "ensemble_predictions.csv")
    parser.add_argument("--metrics-json", type=Path, default=output_root / "ensemble_metrics.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--submission-only", action="store_true")
    parser.add_argument("--xshape-specialist-artifact", type=Path, default=None)
    return parser.parse_args()


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
                writer.writerow([pig_id, *[float(x) for x in prob_row], *[int(x) for x in pred_row]])


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_thresholds(config: dict[str, Any], thresholds_json: Path | None) -> tuple[list[float], dict[str, Any] | None]:
    if thresholds_json is None:
        thresholds = [float(config["inference"]["thresholds"][label]) for label in LABELS]
        return thresholds, None
    with thresholds_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [float(x) for x in payload["thresholds"]], payload.get("postprocess")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)
    rows = load_rows_for_index(args.index_path, args.limit)
    loader = build_loader(config, args.index_path, args.limit)
    thresholds, postprocess_params = load_thresholds(config, args.thresholds_json)

    pig_ids_reference: list[str] | None = None
    targets_reference: np.ndarray | None = None
    has_target_reference: np.ndarray | None = None
    probs_per_checkpoint: list[np.ndarray] = []

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

        pig_ids, targets, main_probs, has_target, aux_probs = predict(
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
        )
        probs = maybe_apply_geometry_fusion(
            rows,
            probs,
            config,
            postprocess_params=postprocess_params,
            skip_labels=specialist_skip_labels if specialist_skip_labels else None,
        )
        probs_per_checkpoint.append(probs)

        if pig_ids_reference is None:
            pig_ids_reference = pig_ids
            targets_reference = targets
            has_target_reference = has_target
        else:
            if pig_ids_reference != pig_ids:
                raise ValueError("Checkpoint predictions are misaligned by pig_id.")

    if not probs_per_checkpoint:
        raise ValueError("No checkpoints were provided.")

    mean_probs = np.mean(np.stack(probs_per_checkpoint, axis=0), axis=0)
    if args.xshape_specialist_artifact is not None:
        artifact_payload = torch.load(args.xshape_specialist_artifact, map_location="cpu")
        _, rear_embeddings, _ = extract_ensemble_rear_embeddings(
            config=config,
            rows=prepare_raw_nometa_rows(rows),
            checkpoints=list(args.checkpoints),
        )
        specialist_scores = compute_xshape_specialist_scores(rear_embeddings, artifact_payload)
        xshape_idx = LABELS.index("x_shape")
        fusion_alpha = float(artifact_payload.get("fusion_alpha", 0.8))
        mean_probs[:, xshape_idx] = np.maximum(mean_probs[:, xshape_idx], fusion_alpha * specialist_scores)
        thresholds[xshape_idx] = float(artifact_payload.get("decision_threshold", thresholds[xshape_idx]))
    preds = apply_thresholds(mean_probs, thresholds)

    write_predictions(
        pig_ids_reference or [],
        mean_probs,
        preds,
        args.output_csv,
        submission_only=bool(args.submission_only),
    )

    metrics = None
    if has_target_reference is not None and bool(has_target_reference.all()):
        metrics = compute_macro_f1(targets_reference, mean_probs, thresholds=thresholds)
        write_metrics(metrics, args.metrics_json)

    print(f"Predictions saved to: {args.output_csv}")
    if metrics is not None:
        print(f"Metrics saved to: {args.metrics_json}")
        print(metrics)
    else:
        print("Targets are unavailable for this index, so metrics were not computed.")


if __name__ == "__main__":
    main()
