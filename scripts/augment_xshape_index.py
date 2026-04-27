"""Expand rare x_shape positives in train index before internal split."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--target-xshape-count",
        type=int,
        default=12,
        help="Target number of x_shape-positive rows after expansion.",
    )
    parser.add_argument(
        "--max-copies-per-row",
        type=int,
        default=24,
        help="Safety cap for number of synthetic copies per source row.",
    )
    return parser.parse_args()


def read_rows(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def with_default_tracking_fields(row: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    updated["source_pig_id"] = str(row.get("source_pig_id") or row["pig_id"])
    updated["is_xshape_augmented"] = str(row.get("is_xshape_augmented") or "0")
    return updated


def expand_rows(
    rows: list[dict[str, str]],
    target_xshape_count: int,
    max_copies_per_row: int,
) -> list[dict[str, str]]:
    base_rows = [with_default_tracking_fields(row) for row in rows]
    xshape_rows = [row for row in base_rows if int(row.get("x_shape", "0") or 0) == 1]
    current_positive_count = len(xshape_rows)
    if current_positive_count == 0 or current_positive_count >= target_xshape_count:
        return base_rows

    needed = target_xshape_count - current_positive_count
    copies_per_row = min(max_copies_per_row, max(1, math.ceil(needed / current_positive_count)))
    synthetic_rows: list[dict[str, str]] = []
    generated = 0
    for row in xshape_rows:
        for copy_idx in range(copies_per_row):
            if generated >= needed:
                break
            new_row = dict(row)
            source_pig_id = row["source_pig_id"]
            new_row["pig_id"] = f"{source_pig_id}__xaug{copy_idx + 1:02d}"
            new_row["source_pig_id"] = source_pig_id
            new_row["is_xshape_augmented"] = "1"
            synthetic_rows.append(new_row)
            generated += 1
        if generated >= needed:
            break
    return base_rows + synthetic_rows


def main() -> None:
    args = parse_args()
    rows = read_rows(args.index_path)
    expanded = expand_rows(
        rows=rows,
        target_xshape_count=int(args.target_xshape_count),
        max_copies_per_row=int(args.max_copies_per_row),
    )
    write_rows(expanded, args.output_path)

    xshape_before = sum(int(row.get("x_shape", "0") or 0) for row in rows)
    xshape_after = sum(int(row.get("x_shape", "0") or 0) for row in expanded)
    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(expanded)}")
    print(f"x_shape positives before: {xshape_before}")
    print(f"x_shape positives after: {xshape_after}")
    print(f"Saved expanded index: {args.output_path}")


if __name__ == "__main__":
    main()
