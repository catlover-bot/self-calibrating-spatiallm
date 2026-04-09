#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/eval_pack/latest}"
MANIFEST_PATH="${MANIFEST_PATH:-configs/eval_pack/small_eval_pack.json}"

mkdir -p "$EVAL_OUTPUT_DIR"

echo "[1/6] Running environment check..."
PYTHONPATH=src "$PYTHON_BIN" scripts/check_environment.py --format json > "$EVAL_OUTPUT_DIR/environment_report_precheck.json"

echo "[2/6] Confirming true calibration_v1 support..."
export EVAL_OUTPUT_DIR
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

report_path = Path(os.environ["EVAL_OUTPUT_DIR"]) / "environment_report_precheck.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
readiness = report.get("readiness", {})
ready = bool(readiness.get("v0_v1_method_comparison_ready"))
if not ready:
    raise SystemExit(
        "Environment is not ready for a trustworthy v0-v1 method comparison. "
        "Install extras with `pip install -e \".[calibration_v1,test]\"`, rerun check, then rerun this workflow."
    )
PY

echo "[3/6] Running evaluation pack..."
PYTHONPATH=src "$PYTHON_BIN" scripts/run_eval_pack.py --manifest "$MANIFEST_PATH" --output-dir "$EVAL_OUTPUT_DIR"

echo "[4/6] Running failure analysis..."
PYTHONPATH=src "$PYTHON_BIN" scripts/run_failure_analysis.py \
  --evaluation-report "$EVAL_OUTPUT_DIR/evaluation_report.json" \
  --output-dir "$EVAL_OUTPUT_DIR"

echo "[5/6] Running dedicated post-true-v1 analysis..."
PYTHONPATH=src "$PYTHON_BIN" scripts/run_post_true_v1_analysis.py \
  --evaluation-report "$EVAL_OUTPUT_DIR/evaluation_report.json" \
  --output-dir "$EVAL_OUTPUT_DIR"

echo "[6/6] Completed. Inspect these first:"
echo "- $EVAL_OUTPUT_DIR/researcher_summary.md"
echo "- $EVAL_OUTPUT_DIR/trustworthy_comparison_status.md"
echo "- $EVAL_OUTPUT_DIR/next_improvement_decision.md"
