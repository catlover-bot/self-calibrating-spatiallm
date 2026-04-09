# Environment Validation

Use environment checks before claiming calibration-v1 results.

## Why

`plane_aware_v1` requires `numpy` for true plane-aware execution.  
If `numpy` is unavailable, v1 gracefully falls back and should not be treated as true v1 evidence.

## Dependency Setup

Core:

```bash
pip install -e .
```

True v1 calibration support:

```bash
pip install -e ".[calibration_v1]"
```

Tests:

```bash
pip install -e ".[test]"
```

Combined true-v1 + test install:

```bash
pip install -e ".[calibration_v1,test]"
```

Helper setup script:

```bash
bash scripts/setup_true_v1_env.sh
```

## Check Command

```bash
PYTHONPATH=src python scripts/check_environment.py --format text
```

Or:

```bash
PYTHONPATH=src python -m self_calibrating_spatiallm.cli check-env --format json
```

The report covers:

- calibration v1 true-path support
- point cloud loading backend support
- external SpatialLM adapter availability

## How To Verify v1 Really Ran

Inspect:

- `03_calibration_result.json`:
  - `metadata.execution.true_v1_execution`
  - `metadata.execution.fallback_used`
  - `metadata.execution.partial_calibration_applied`
  - `metadata.execution.fallback_reason`
  - `metadata.diagnostics.up_axis_selection`
  - `metadata.diagnostics.up_axis_guardrail`
  - `metadata.diagnostics.horizontal_guardrail`
  - `metadata.diagnostics.reliability_breakdown`
  - `metadata.diagnostics.wall_candidate_ranking`
  - `metadata.diagnostics.horizontal_evidence`
- `run_manifest.json`:
  - `calibration_execution`

For evaluation packs, inspect `evaluation_report.json -> v1_execution_summary`.

## End-to-End Commands (Exact Sequence)

1. Install deps:

```bash
pip install -e ".[calibration_v1,test]"
```

2. Check environment:

```bash
PYTHONPATH=src python scripts/check_environment.py --format text
```

3. Confirm:

- `true_calibration_v1_ready: True`
- `v0_v1_method_comparison_ready: True`

4. Run eval pack:

```bash
PYTHONPATH=src python scripts/run_eval_pack.py \
  --manifest configs/eval_pack/small_eval_pack.json \
  --output-dir outputs/eval_pack/latest
```

5. Run failure analysis:

```bash
PYTHONPATH=src python scripts/run_failure_analysis.py \
  --evaluation-report outputs/eval_pack/latest/evaluation_report.json \
  --output-dir outputs/eval_pack/latest
```

6. Inspect first:

- `outputs/eval_pack/latest/readiness_summary.md`
- `outputs/eval_pack/latest/researcher_summary.md`
- `outputs/eval_pack/latest/trustworthy_comparison_status.md`
- `outputs/eval_pack/latest/next_improvement_decision.md`

## Fallback-Heavy Results

If fallback counts are high, treat v0-v1 deltas as inconclusive for plane-aware calibration quality.  
Prioritize scenes with true v1 execution when making method claims.

If `num_true_v1_execution == 0`, do not make a v0-v1 method claim.
If `trustworthy_comparison_status.conclusions_provisional == true`, treat conclusions as provisional.
