"""Fit healthy-vs-positive prototype specialists over ensemble embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.env import get_output_root, load_local_env
from oinkvision.infer import load_config
from oinkvision.prototype_specialist import (
    extract_ensemble_multiview_embeddings,
    fit_prototype_specialist,
    load_rows_and_labels,
)


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, default=output_root / "index" / "train_index.csv")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-artifact", type=Path, default=output_root / "prototype_specialist.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows, labels_matrix = load_rows_and_labels(str(args.index_path))
    _, all_embeddings, rear_embeddings, base_probs = extract_ensemble_multiview_embeddings(
        config=config,
        rows=rows,
        checkpoints=[str(path) for path in args.checkpoints],
    )
    payload = fit_prototype_specialist(
        all_embeddings=all_embeddings,
        rear_embeddings=rear_embeddings,
        labels_matrix=labels_matrix,
        base_probs=base_probs,
    )
    payload.update(
        {
            "config": str(args.config),
            "index_path": str(args.index_path),
            "checkpoints": [str(path) for path in args.checkpoints],
        }
    )
    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_artifact)
    print(f"Prototype specialist artifact saved to: {args.output_artifact}")
    print({label: data.get("train_metrics", {}) for label, data in payload.get("labels", {}).items()})


if __name__ == "__main__":
    main()
