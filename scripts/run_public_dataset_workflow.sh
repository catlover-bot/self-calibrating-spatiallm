#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST_PATH="${MANIFEST_PATH:-configs/public_datasets/arkitscenes/generated/arkitscenes_subset_manifest.json}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/eval_pack/public_medium_latest}"

mkdir -p "$EVAL_OUTPUT_DIR"

echo "[1/3] Running evaluation pack on public-dataset subset..."
PYTHONPATH=src "$PYTHON_BIN" scripts/run_eval_pack.py \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$EVAL_OUTPUT_DIR"

echo "[2/3] Running failure analysis..."
PYTHONPATH=src "$PYTHON_BIN" scripts/run_failure_analysis.py \
  --evaluation-report "$EVAL_OUTPUT_DIR/evaluation_report.json" \
  --output-dir "$EVAL_OUTPUT_DIR"

echo "[3/3] Running post-run analysis..."
PYTHONPATH=src "$PYTHON_BIN" scripts/run_post_true_v1_analysis.py \
  --evaluation-report "$EVAL_OUTPUT_DIR/evaluation_report.json" \
  --output-dir "$EVAL_OUTPUT_DIR"

echo "Completed public-dataset workflow. Inspect these first:"
echo "- $EVAL_OUTPUT_DIR/researcher_summary.md"
echo "- $EVAL_OUTPUT_DIR/trustworthy_comparison_status.md"
echo "- $EVAL_OUTPUT_DIR/scene_level_delta_report.md"
echo "- $EVAL_OUTPUT_DIR/external_propagation_summary.md"
