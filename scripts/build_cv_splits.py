"""Build reproducible K-fold train/valid CSV splits."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.dataset import load_index
from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, default=output_root / "index" / "train_index.csv")
    parser.add_argument("--output-dir", type=Path, default=output_root / "cv_splits")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_index(args.index_path)
    labels = np.array([f"{r['bad_posture']}{r['bumps']}{r['soft_pastern']}{r['x_shape']}" for r in rows])
    indices = np.arange(len(rows))

    try:
        splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(indices, labels)
    except ValueError:
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(indices)

    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter):
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        train_rows = [rows[i] for i in train_idx]
        valid_rows = [rows[i] for i in valid_idx]
        write_rows(train_rows, fold_dir / "train.csv")
        write_rows(valid_rows, fold_dir / "valid.csv")
        print(f"fold_{fold_idx}: train={len(train_rows)} valid={len(valid_rows)}")


if __name__ == "__main__":
    main()
