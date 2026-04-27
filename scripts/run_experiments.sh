#!/usr/bin/env bash

set -euo pipefail

# Reproducible end-to-end OOF experiments runner.
# Usage examples:
#   bash scripts/run_experiments.sh
#   bash scripts/run_experiments.sh --config configs/baseline_server_v3.yaml --epochs 18 --seeds "42 52 62"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="configs/baseline_server_v3.yaml"
INDEX_PATH="outputs/index/train_index.csv"
BASE_OUTPUT_DIR="outputs/exp_oof"
N_SPLITS=5
N_REPEATS=3
EPOCHS=""
SEEDS="42 52 62"
REBUILD_INDEX=1
PREEXTRACT_CACHE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --index-path)
      INDEX_PATH="$2"
      shift 2
      ;;
    --output-dir)
      BASE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --n-splits)
      N_SPLITS="$2"
      shift 2
      ;;
    --n-repeats)
      N_REPEATS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --skip-index)
      REBUILD_INDEX=0
      shift
      ;;
    --preextract-cache)
      PREEXTRACT_CACHE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "=== OOF experiments runner ==="
echo "root: ${ROOT_DIR}"
echo "python: ${PYTHON_BIN}"
echo "config: ${CONFIG_PATH}"
echo "index: ${INDEX_PATH}"
echo "out: ${BASE_OUTPUT_DIR}"
echo "n_splits: ${N_SPLITS}"
echo "n_repeats: ${N_REPEATS}"
echo "epochs override: ${EPOCHS:-<config default>}"
echo "seeds: ${SEEDS}"

if [[ "${REBUILD_INDEX}" -eq 1 ]]; then
  echo ""
  echo "[1/4] Building train index..."
  "${PYTHON_BIN}" scripts/build_index.py
else
  echo ""
  echo "[1/4] Skipping index rebuild (--skip-index)"
fi

if [[ "${PREEXTRACT_CACHE}" -eq 1 ]]; then
  echo ""
  echo "[2/4] Pre-extracting annotated frame cache..."
  "${PYTHON_BIN}" scripts/preextract_annotated_frames.py \
    --config "${CONFIG_PATH}" \
    --index-path "${INDEX_PATH}"
else
  echo ""
  echo "[2/4] Skipping frame cache pre-extraction (use --preextract-cache to enable)"
fi

echo ""
echo "[3/4] Running repeated OOF CV experiments..."
for seed in ${SEEDS}; do
  RUN_DIR="${BASE_OUTPUT_DIR}/seed_${seed}"
  mkdir -p "${RUN_DIR}"
  CMD=(
    "${PYTHON_BIN}" scripts/run_oof_cv.py
    --config "${CONFIG_PATH}"
    --index-path "${INDEX_PATH}"
    --output-dir "${RUN_DIR}"
    --n-splits "${N_SPLITS}"
    --n-repeats "${N_REPEATS}"
    --seed "${seed}"
  )
  if [[ -n "${EPOCHS}" ]]; then
    CMD+=(--epochs "${EPOCHS}")
  fi
  echo "Running seed=${seed} -> ${RUN_DIR}"
  "${CMD[@]}"
done

echo ""
echo "[4/4] Aggregating experiment summary..."
BASE_OUTPUT_DIR_ENV="${BASE_OUTPUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
import statistics

base = Path(os.environ["BASE_OUTPUT_DIR_ENV"])
if not base.exists():
    raise SystemExit(f"No experiments found at {base}")

runs = sorted([p for p in base.glob("seed_*") if p.is_dir()])
if not runs:
    raise SystemExit("No seed_* directories found")

rows = []
for run in runs:
    summary_path = run / "oof_summary.json"
    if not summary_path.exists():
        continue
    with summary_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    rows.append(
        {
            "seed": run.name.replace("seed_", ""),
            "macro_default_05": float(d["default_metrics_05"]["macro_f1"]),
            "macro_tuned_oof": float(d["tuned_oof_metrics"]["macro_f1"]),
            "thresholds": d["tuned_oof_metrics"]["thresholds"],
        }
    )

if not rows:
    raise SystemExit("No valid oof_summary.json files found")

print("\nPer-seed results:")
for r in rows:
    print(
        f"- seed={r['seed']}: macro@0.5={r['macro_default_05']:.4f}, "
        f"macro@tuned_oof={r['macro_tuned_oof']:.4f}, thresholds={r['thresholds']}"
    )

vals_default = [r["macro_default_05"] for r in rows]
vals_tuned = [r["macro_tuned_oof"] for r in rows]
print("\nOverall:")
print(
    f"- default@0.5 mean={statistics.mean(vals_default):.4f} std={statistics.pstdev(vals_default):.4f}"
)
print(
    f"- tuned_oof mean={statistics.mean(vals_tuned):.4f} std={statistics.pstdev(vals_tuned):.4f}"
)
PY

echo ""
echo "Done. Detailed artifacts are in: ${BASE_OUTPUT_DIR}/seed_*/oof_summary.json"
