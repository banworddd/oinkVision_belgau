"""Build animal-level index from raw video folders without labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oinkvision.env import get_data_root, get_output_root, load_local_env
from oinkvision.indexing import build_records_from_raw_dir, records_to_rows, validate_record


def parse_args() -> argparse.Namespace:
    load_local_env()
    data_root = get_data_root()
    output_root = get_output_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["val", "test"], required=True)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=output_root / "index")
    return parser.parse_args()


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir if args.raw_dir is not None else get_data_root() / args.split / "raw"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records_from_raw_dir(raw_dir)
    rows = records_to_rows(records)

    errors: list[str] = []
    for record in records:
        errors.extend(validate_record(record))
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    csv_path = args.output_dir / f"{args.split}_index.csv"
    json_path = args.output_dir / f"{args.split}_index.json"
    write_csv(rows, csv_path)
    write_json(rows, json_path)

    print(f"Built {len(rows)} unlabeled records for {args.split}")
    print(f"CSV index: {csv_path}")
    print(f"JSON index: {json_path}")


if __name__ == "__main__":
    main()
