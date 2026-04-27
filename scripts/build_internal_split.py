"""Create reproducible internal train/valid CSV splits from the full index."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.train import load_config, split_rows
from oinkvision.dataset import load_index
from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=output_root / "index" / "train_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "splits",
    )
    return parser.parse_args()


def write_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = load_index(args.index_path)
    train_rows, valid_rows = split_rows(
        rows,
        seed=int(config["seed"]),
        valid_size=float(config["train"]["valid_size"]),
    )

    train_path = args.output_dir / "train_split.csv"
    valid_path = args.output_dir / "valid_split.csv"
    write_rows(train_rows, train_path)
    write_rows(valid_rows, valid_path)

    print(f"Train split: {train_path} ({len(train_rows)} rows)")
    print(f"Valid split: {valid_path} ({len(valid_rows)} rows)")


if __name__ == "__main__":
    main()
