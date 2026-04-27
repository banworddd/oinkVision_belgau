"""Evaluate a trained checkpoint on train index in replay modes.

Useful for diagnosing overfitting and domain gap:
- standard mode: as configured
- raw-video mode: ignore annotation frame sampling
- no-metadata mode: zero front-leg metadata features
"""

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
    maybe_apply_geometry_fusion,
    maybe_apply_specialist_fusion,
    predict,
)
from oinkvision.metrics import apply_thresholds, compute_macro_f1
from oinkvision.model import build_model
from oinkvision.train import build_aggregation_spec, choose_device, load_config


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, default=output_root / "index" / "train_index.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=output_root / "train_replay_metrics.json")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["raw_nometa", "configured", "both"],
        default="raw_nometa",
        help="raw_nometa: no metadata + raw video only; configured: as config; both: run both and compare",
    )
    return parser.parse_args()


def prepare_rows(rows: list[dict], force_raw_video: bool, disable_metadata: bool) -> list[dict]:
    prepared: list[dict] = []
    for row in rows:
        item = dict(row)
        if force_raw_video:
            item["annotation_path"] = ""
        if disable_metadata:
            for field in FRONT_META_FIELDS:
                item[field] = 0
        prepared.append(item)
    return prepared


def build_loader_from_rows(config: dict, rows: list[dict]) -> DataLoader:
    dataset = PigVideoDataset(
        index_path="",
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
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )


def load_thresholds(config: dict, thresholds_json: Path | None) -> list[float]:
    if thresholds_json is None:
        return [float(config["inference"]["thresholds"][label]) for label in LABELS]
    with thresholds_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [float(x) for x in payload["thresholds"]]


def build_confusion(targets: np.ndarray, probs: np.ndarray, thresholds: list[float]) -> dict[str, dict[str, int]]:
    preds = apply_thresholds(probs, thresholds)
    result: dict[str, dict[str, int]] = {}
    for idx, label in enumerate(LABELS):
        y_true = targets[:, idx].astype(np.int64)
        y_pred = preds[:, idx].astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        result[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return result


def run_eval(
    config: dict,
    model: torch.nn.Module,
    device: torch.device,
    aggregation_spec: dict,
    thresholds: list[float],
    rows_source: list[dict],
    force_raw_video: bool,
    disable_metadata: bool,
) -> dict:
    rows = prepare_rows(rows_source, force_raw_video=force_raw_video, disable_metadata=disable_metadata)
    loader = build_loader_from_rows(config, rows)

    _, targets, main_probs, has_target, aux_probs = predict(
        model,
        loader,
        device,
        aggregation_spec=aggregation_spec,
    )
    if not bool(has_target.all()):
        raise ValueError("Train replay evaluation requires labeled rows.")

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
    metrics = compute_macro_f1(targets, probs, thresholds=thresholds)
    confusion = build_confusion(targets, probs, thresholds=thresholds)
    return {
        "force_raw_video": force_raw_video,
        "disable_metadata": disable_metadata,
        "metrics": metrics,
        "confusion": confusion,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device()
    aggregation_spec = build_aggregation_spec(config, device)
    thresholds = load_thresholds(config, args.thresholds_json)

    rows = load_index(args.index_path)

    model = build_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    payload: dict[str, object] = {
        "config": str(args.config),
        "index_path": str(args.index_path),
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
    }
    if args.mode == "configured":
        payload["result"] = run_eval(
            config, model, device, aggregation_spec, thresholds, rows, force_raw_video=False, disable_metadata=False
        )
    elif args.mode == "both":
        configured = run_eval(
            config, model, device, aggregation_spec, thresholds, rows, force_raw_video=False, disable_metadata=False
        )
        raw_nometa = run_eval(
            config, model, device, aggregation_spec, thresholds, rows, force_raw_video=True, disable_metadata=True
        )
        payload["configured"] = configured
        payload["raw_nometa"] = raw_nometa
        payload["delta_macro_f1_raw_minus_configured"] = (
            float(raw_nometa["metrics"]["macro_f1"]) - float(configured["metrics"]["macro_f1"])  # type: ignore[index]
        )
    else:
        payload["result"] = run_eval(
            config, model, device, aggregation_spec, thresholds, rows, force_raw_video=True, disable_metadata=True
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Train replay metrics saved to: {args.output_json}")
    print(payload)


if __name__ == "__main__":
    main()
