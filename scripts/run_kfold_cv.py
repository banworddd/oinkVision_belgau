"""Run K-fold training/evaluation for the main baseline config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, default=output_root / "index" / "train_index.csv")
    parser.add_argument("--splits-dir", type=Path, default=output_root / "cv_splits")
    parser.add_argument("--runs-dir", type=Path, default=output_root / "cv_runs")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rebuild-splits", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable

    if args.rebuild_splits or not args.splits_dir.exists():
        run_command(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "build_cv_splits.py"),
                "--index-path",
                str(args.index_path),
                "--output-dir",
                str(args.splits_dir),
                "--n-splits",
                str(args.n_splits),
                "--seed",
                str(args.seed),
            ]
        )

    fold_dirs = sorted(path for path in args.splits_dir.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {args.splits_dir}")

    args.runs_dir.mkdir(parents=True, exist_ok=True)

    fold_scores = []
    fold_payloads = []

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        run_dir = args.runs_dir / fold_name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            python,
            str(PROJECT_ROOT / "src" / "oinkvision" / "train.py"),
            "--config",
            str(args.config),
            "--train-index",
            str(fold_dir / "train.csv"),
            "--valid-index",
            str(fold_dir / "valid.csv"),
            "--output-dir",
            str(run_dir),
        ]
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        run_command(train_cmd)

        run_command(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "tune_thresholds.py"),
                "--config",
                str(args.config),
                "--index-path",
                str(fold_dir / "valid.csv"),
                "--checkpoint",
                str(run_dir / "best_model.pt"),
                "--output-json",
                str(run_dir / "tuned_thresholds.json"),
            ]
        )

        run_command(
            [
                python,
                str(PROJECT_ROOT / "src" / "oinkvision" / "infer.py"),
                "--config",
                str(args.config),
                "--index-path",
                str(fold_dir / "valid.csv"),
                "--checkpoint",
                str(run_dir / "best_model.pt"),
                "--thresholds-json",
                str(run_dir / "tuned_thresholds.json"),
                "--output-csv",
                str(run_dir / "valid_tuned_predictions.csv"),
                "--metrics-json",
                str(run_dir / "valid_tuned_metrics.json"),
            ]
        )

        metrics_path = run_dir / "valid_tuned_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        fold_scores.append(float(metrics["macro_f1"]))
        fold_payloads.append({"fold": fold_name, **metrics})

    summary = {
        "config": str(args.config),
        "n_folds": len(fold_dirs),
        "macro_f1_mean": float(np.mean(fold_scores)),
        "macro_f1_std": float(np.std(fold_scores)),
        "folds": fold_payloads,
    }
    summary_path = args.runs_dir / "cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"CV summary saved to: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
