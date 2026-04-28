"""Tune per-class thresholds on a validation split using a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from oinkvision.constants import LABELS, get_active_labels
from oinkvision.infer import build_loader, load_config, load_rows_for_index, maybe_apply_geometry_fusion, maybe_apply_specialist_fusion, predict
from oinkvision.metrics import compute_macro_f1, optimize_thresholds
from oinkvision.model import build_model
from oinkvision.train import build_aggregation_spec, choose_device
from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=output_root / "tuned_thresholds.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    active_labels = get_active_labels(config)
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)

    model = build_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=False)
    print(
        {
            "checkpoint": str(args.checkpoint),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }
    )

    rows = load_rows_for_index(args.index_path, limit=None)
    loader = build_loader(config, args.index_path, limit=None)
    _, targets, main_probs, has_target, aux_probs = predict(
        model,
        loader,
        device,
        aggregation_spec=aggregation_spec,
    )
    if not bool(has_target.all()):
        raise ValueError("Threshold tuning requires targets, but the provided index has unlabeled rows.")

    probs, specialist_skip_labels = maybe_apply_specialist_fusion(
        rows=rows,
        main_probs=main_probs,
        aux_probs=aux_probs,
        config=config,
        active_labels=active_labels,
    )
    probs = maybe_apply_geometry_fusion(
        rows,
        probs,
        config,
        active_labels=active_labels,
        skip_labels=specialist_skip_labels if specialist_skip_labels else None,
    )

    target_indices = [LABELS.index(label) for label in active_labels]
    targets = targets[:, target_indices]
    base_thresholds = [float(config["inference"]["thresholds"][label]) for label in active_labels]
    base_metrics = compute_macro_f1(targets, probs, thresholds=base_thresholds, labels=active_labels)
    tuned_metrics = optimize_thresholds(targets, probs, labels=active_labels)
    supports = tuned_metrics.get("per_class_support", {})
    fallback_cfg = dict(config.get("inference", {}).get("threshold_fallbacks", {}))
    min_support = int(fallback_cfg.get("min_support", 1))
    fallback_thresholds_cfg = dict(fallback_cfg.get("thresholds", {}))
    guarded_thresholds = list(tuned_metrics["thresholds"])
    for idx, label in enumerate(active_labels):
        if int(supports.get(label, 0)) < min_support and label in fallback_thresholds_cfg:
            guarded_thresholds[idx] = float(fallback_thresholds_cfg[label])
    guarded_metrics = compute_macro_f1(targets, probs, thresholds=guarded_thresholds, labels=active_labels)
    payload = {
        "base_metrics": base_metrics,
        "tuned_metrics": tuned_metrics,
        "guarded_metrics": guarded_metrics,
        "thresholds": guarded_thresholds,
        "threshold_fallback": {
            "min_support": min_support,
            "configured_thresholds": fallback_thresholds_cfg,
            "applied_thresholds": guarded_thresholds,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Thresholds saved to: {args.output_json}")
    print(payload)


if __name__ == "__main__":
    main()
