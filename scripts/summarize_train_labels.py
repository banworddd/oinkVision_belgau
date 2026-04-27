"""Summarize train labels directly from annotation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.constants import LABELS
from oinkvision.env import get_data_root, get_output_root, load_local_env


def parse_args() -> argparse.Namespace:
    load_local_env()
    data_root = get_data_root()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=data_root / "train" / "annotation",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=output_root / "train_label_summary.csv",
    )
    return parser.parse_args()


def load_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_rows(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    annotation_paths = sorted(args.annotation_dir.glob("pig_*.json"))
    if not annotation_paths:
        raise ValueError(f"No annotation JSON files found in {args.annotation_dir}")

    label_counts = {label: 0 for label in LABELS}
    combo_counts: Counter[str] = Counter()
    exported_rows: list[dict[str, str]] = []

    for annotation_path in annotation_paths:
        data = load_annotation(annotation_path)
        pig_id = str(data["pig_id"])
        target = data["target"]

        positive_labels = [label for label in LABELS if int(target[label]) == 1]
        for label in positive_labels:
            label_counts[label] += 1

        disease_list = ", ".join(positive_labels) if positive_labels else "healthy"
        combo_key = " + ".join(positive_labels) if positive_labels else "healthy"
        combo_counts[combo_key] += 1

        exported_rows.append(
            {
                "pig_id": pig_id,
                "annotation_file": annotation_path.name,
                "disease_list": disease_list,
                **{label: str(int(target[label])) for label in LABELS},
            }
        )

    write_rows(exported_rows, args.output_csv)

    print("=== Train label summary (from annotation JSON) ===")
    print(f"num_pigs: {len(exported_rows)}")
    print("")
    print("Per-class positives:")
    for label in LABELS:
        positives = label_counts[label]
        negatives = len(exported_rows) - positives
        print(f"- {label}: positive={positives}, negative={negatives}")

    print("")
    print("Label combinations:")
    for combo, count in combo_counts.most_common():
        print(f"- {combo}: {count}")

    print("")
    print(f"Per-pig table saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
