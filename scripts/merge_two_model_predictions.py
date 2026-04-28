"""Merge a 3-class main prediction file with an x_shape-only prediction file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-csv", type=Path, required=True)
    parser.add_argument("--xshape-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def main() -> None:
    args = parse_args()
    main_rows = load_rows(args.main_csv)
    xshape_rows = load_rows(args.xshape_csv)
    shared_ids = list(main_rows.keys())
    missing = [pig_id for pig_id in shared_ids if pig_id not in xshape_rows]
    if missing:
        raise ValueError(f"x_shape CSV is missing ids: {missing[:5]}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "bad_posture", "bumps", "soft_pastern", "x_shape"])
        for pig_id in shared_ids:
            writer.writerow(
                [
                    pig_id,
                    int(main_rows[pig_id]["bad_posture"]),
                    int(main_rows[pig_id]["bumps"]),
                    int(main_rows[pig_id]["soft_pastern"]),
                    int(xshape_rows[pig_id]["x_shape"]),
                ]
            )

    print(f"Merged submission saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
