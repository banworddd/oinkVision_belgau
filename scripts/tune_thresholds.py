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

from oinkvision.infer import build_loader, load_config, predict
from oinkvision.metrics import compute_macro_f1, optimize_thresholds
from oinkvision.model import build_model
from oinkvision.train import choose_device
from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_laptop.yaml")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=output_root / "tuned_thresholds.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device()

    model = build_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    loader = build_loader(config, args.index_path, limit=None)
    _, targets, probs, has_target = predict(model, loader, device)
    if not bool(has_target.all()):
        raise ValueError("Threshold tuning requires targets, but the provided index has unlabeled rows.")

    base_thresholds = [float(config["inference"]["thresholds"][k]) for k in ["bad_posture", "bumps", "soft_pastern", "x_shape"]]
    base_metrics = compute_macro_f1(targets, probs, thresholds=base_thresholds)
    tuned_metrics = optimize_thresholds(targets, probs)
    payload = {
        "base_metrics": base_metrics,
        "tuned_metrics": tuned_metrics,
        "thresholds": tuned_metrics["thresholds"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Thresholds saved to: {args.output_json}")
    print(payload)


if __name__ == "__main__":
    main()
