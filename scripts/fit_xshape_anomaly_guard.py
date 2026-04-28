"""Fit a conservative rear-embedding anomaly guard for x_shape."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.dataset import load_index
from oinkvision.env import get_output_root, load_local_env
from oinkvision.infer import load_config
from oinkvision.xshape_anomaly import fit_xshape_anomaly_guard


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-artifact", type=Path, default=output_root / "xshape_anomaly_guard.pt")
    parser.add_argument("--output-json", type=Path, default=output_root / "xshape_anomaly_guard_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = load_index(args.index_path)
    artifact = fit_xshape_anomaly_guard(
        config=config,
        rows=rows,
        checkpoints=[str(path) for path in args.checkpoints],
    )
    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output_artifact)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(artifact["train_summary"], f, ensure_ascii=False, indent=2)
    print(f"x_shape anomaly guard saved to: {args.output_artifact}")
    print(artifact["train_summary"])


if __name__ == "__main__":
    main()
