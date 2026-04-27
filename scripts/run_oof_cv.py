"""Run repeated rare-aware OOF CV and report honest macro-F1 on train data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.constants import LABELS
from oinkvision.dataset import load_index
from oinkvision.metrics import compute_macro_f1, optimize_thresholds
from oinkvision.env import get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline_server_v3.yaml")
    parser.add_argument("--index-path", type=Path, default=output_root / "index" / "train_index.csv")
    parser.add_argument("--output-dir", type=Path, default=output_root / "oof_cv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_probabilities(pred_csv: Path) -> tuple[list[str], np.ndarray]:
    rows = list(csv.DictReader(pred_csv.open("r", encoding="utf-8", newline="")))
    pig_ids = [str(row["id"]) for row in rows]
    probs = np.array(
        [[float(row[f"{label}_prob"]) for label in LABELS] for row in rows],
        dtype=np.float32,
    )
    return pig_ids, probs


def make_rare_aware_splits(rows: list[dict[str, Any]], n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(rows) < n_splits:
        raise ValueError("Not enough rows for requested n_splits")

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    pos_counts = {
        label: max(sum(int(rows[idx][label]) for idx in indices), 1)
        for label in LABELS
    }
    rarity_scores = {}
    for idx in indices:
        rarity = 0.0
        for label in LABELS:
            if int(rows[idx][label]) == 1:
                rarity += 1.0 / pos_counts[label]
        rarity_scores[idx] = rarity
    indices.sort(key=lambda idx: (rarity_scores[idx], rng.random()), reverse=True)

    desired_size = len(rows) / n_splits
    desired_pos = {label: pos_counts[label] / n_splits for label in LABELS}
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_pos = [{label: 0 for label in LABELS} for _ in range(n_splits)]

    for idx in indices:
        best_fold = 0
        best_score = float("inf")
        sample_labels = {label: int(rows[idx][label]) for label in LABELS}
        for fold_id in range(n_splits):
            size_after = len(fold_indices[fold_id]) + 1
            size_penalty = abs(size_after - desired_size) * 0.6
            label_penalty = 0.0
            for label in LABELS:
                after = fold_pos[fold_id][label] + sample_labels[label]
                label_penalty += abs(after - desired_pos[label])
            score = size_penalty + label_penalty
            if score < best_score:
                best_score = score
                best_fold = fold_id
        fold_indices[best_fold].append(idx)
        for label in LABELS:
            fold_pos[best_fold][label] += sample_labels[label]

    splits: list[tuple[list[int], list[int]]] = []
    all_indices = set(range(len(rows)))
    for fold_id in range(n_splits):
        valid_idx = sorted(fold_indices[fold_id])
        train_idx = sorted(all_indices - set(valid_idx))
        splits.append((train_idx, valid_idx))
    return splits


def main() -> None:
    args = parse_args()
    python = sys.executable
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_index(args.index_path)
    rows_by_id = {str(row["pig_id"]): row for row in rows}

    oof_probs: list[np.ndarray] = []
    oof_targets: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []

    for repeat_idx in range(args.n_repeats):
        repeat_seed = args.seed + repeat_idx * 1000
        repeat_dir = args.output_dir / f"repeat_{repeat_idx}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        splits = make_rare_aware_splits(rows, n_splits=args.n_splits, seed=repeat_seed)

        for fold_idx, (train_idx, valid_idx) in enumerate(splits):
            fold_dir = repeat_dir / f"fold_{fold_idx}"
            train_rows = [rows[i] for i in train_idx]
            valid_rows = [rows[i] for i in valid_idx]
            train_csv = fold_dir / "train.csv"
            valid_csv = fold_dir / "valid.csv"
            write_rows(train_rows, train_csv)
            write_rows(valid_rows, valid_csv)

            train_cmd = [
                python,
                str(PROJECT_ROOT / "src" / "oinkvision" / "train.py"),
                "--config",
                str(args.config),
                "--train-index",
                str(train_csv),
                "--valid-index",
                str(valid_csv),
                "--output-dir",
                str(fold_dir),
                "--seed",
                str(repeat_seed + fold_idx),
            ]
            if args.epochs is not None:
                train_cmd.extend(["--epochs", str(args.epochs)])
            run_command(train_cmd)

            pred_csv = fold_dir / "valid_predictions.csv"
            run_command(
                [
                    python,
                    str(PROJECT_ROOT / "src" / "oinkvision" / "infer.py"),
                    "--config",
                    str(args.config),
                    "--index-path",
                    str(valid_csv),
                    "--checkpoint",
                    str(fold_dir / "best_model.pt"),
                    "--output-csv",
                    str(pred_csv),
                    "--metrics-json",
                    str(fold_dir / "valid_metrics_default_thresholds.json"),
                ]
            )

            pig_ids, probs = read_probabilities(pred_csv)
            targets = np.array(
                [[int(rows_by_id[pig_id][label]) for label in LABELS] for pig_id in pig_ids],
                dtype=np.int64,
            )
            oof_probs.append(probs)
            oof_targets.append(targets)

            fold_default = compute_macro_f1(targets, probs, thresholds=[0.5] * len(LABELS))
            fold_metrics.append(
                {
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "macro_f1_default_05": fold_default["macro_f1"],
                    "per_class_default_05": fold_default["per_class_f1"],
                }
            )

    all_probs = np.concatenate(oof_probs, axis=0)
    all_targets = np.concatenate(oof_targets, axis=0)

    default_metrics = compute_macro_f1(all_targets, all_probs, thresholds=[0.5] * len(LABELS))
    tuned_metrics = optimize_thresholds(all_targets, all_probs)

    summary = {
        "config": str(args.config),
        "index_path": str(args.index_path),
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "num_oof_samples": int(all_targets.shape[0]),
        "default_metrics_05": default_metrics,
        "tuned_oof_metrics": tuned_metrics,
        "fold_metrics_default_05": fold_metrics,
    }
    summary_path = args.output_dir / "oof_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    thresholds_path = args.output_dir / "oof_thresholds.json"
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump({"thresholds": tuned_metrics["thresholds"]}, f, ensure_ascii=False, indent=2)

    print(f"OOF summary: {summary_path}")
    print(f"OOF thresholds: {thresholds_path}")
    print({"macro_f1_default_05": default_metrics["macro_f1"], "macro_f1_tuned_oof": tuned_metrics["macro_f1"]})


if __name__ == "__main__":
    main()
