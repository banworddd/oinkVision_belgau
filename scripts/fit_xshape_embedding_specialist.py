"""Fit rear-embedding anomaly specialist for x_shape using ensemble checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.dataset import load_index
from oinkvision.infer import load_config
from oinkvision.xshape_specialist import (
    extract_ensemble_rear_embeddings,
    fit_xshape_specialist,
    prepare_raw_nometa_rows,
)
from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, default=output_root / "index" / "train_index.csv")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-artifact", type=Path, default=output_root / "xshape_embedding_specialist.pt")
    parser.add_argument("--knn-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = prepare_raw_nometa_rows(load_index(args.index_path))
    _, rear_embeddings, base_xshape_probs = extract_ensemble_rear_embeddings(
        config=config,
        rows=rows,
        checkpoints=list(args.checkpoints),
    )
    xshape_labels = torch.tensor([int(row.get("x_shape", 0)) for row in rows], dtype=torch.int64).numpy()
    artifact = fit_xshape_specialist(
        embeddings=rear_embeddings,
        xshape_labels=xshape_labels,
        base_xshape_probs=base_xshape_probs,
        knn_k=int(args.knn_k),
    )
    payload = {
        "config": str(args.config),
        "index_path": str(args.index_path),
        "checkpoints": [str(path) for path in args.checkpoints],
        **artifact.to_payload(),
    }
    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_artifact)
    print(f"x_shape specialist artifact saved to: {args.output_artifact}")
    print(payload["train_metrics"])


if __name__ == "__main__":
    main()
