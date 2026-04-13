# Evaluation Pack Guide

This repository supports small-scale (5-10 scene) evidence-building workflows.

## Annotation Model

Per-scene lightweight annotation JSON supports:

- `scene_id`
- `expected_up_axis`
- `expected_horizontal_axis`
- `room_orientation_hint`
- `expected_scale_hint`
- `expected_door_count`
- `expected_window_count`
- `expected_object_categories`
- `expected_relations` (category-level relations)
- `traversability_labels`
- `notes`

Schema:

- `schemas/scene_annotation.schema.json`

Examples:

- `configs/annotations/*.annotation.json`

## Evaluation Pack Manifest

Manifest format includes:

- `sample_config_path`
- `annotation_path` (optional)
- `source_type`
- `tags`
- `notes`

Schema:

- `schemas/evaluation_pack_manifest.schema.json`

Example:

- `configs/eval_pack/small_eval_pack.json`

## Running

```bash
PYTHONPATH=src python scripts/run_eval_pack.py \
  --manifest configs/eval_pack/small_eval_pack.json \
  --output-dir outputs/eval_pack/latest
```

Then run failure analysis:

```bash
PYTHONPATH=src python scripts/run_failure_analysis.py \
  --evaluation-report outputs/eval_pack/latest/evaluation_report.json \
  --output-dir outputs/eval_pack/latest
```

Optional dedicated post-true-v1 interpretation pass (for already completed eval runs):

```bash
PYTHONPATH=src python scripts/run_post_true_v1_analysis.py \
  --evaluation-report outputs/eval_pack/latest/evaluation_report.json \
  --output-dir outputs/eval_pack/latest
```

Public dataset subset workflow (ARKitScenes-first scaffold):

- `docs/public_datasets.md`
- `docs/language_layer.md`

## Compared Settings

- `no_calibration`
- `calibration_v0`
- `calibration_v1`
- `calibration_v1_plus_repair`
- `mock_generator`
- `external_generator` (when configured)

## Metrics

Calibration:

- up-axis error (degrees)
- horizontal orientation error (degrees, axis-symmetric by default; explicit signed axis labels use directed error)
- up-axis confidence
- horizontal confidence
- overall reliability (mode-aware: full calibration vs safe partial vs degraded fallback)
- manhattan ambiguity
- scale match/error when expected scale hint exists

Structured scene:

- object category presence precision/recall/F1
- door/window count error
- pre-repair violation count

Repair:

- violations before vs after
- fixes applied
- overcorrection flag

Actionable scene:

- simple relation precision/recall/F1
- traversability accuracy when labels exist

## Outputs

- `evaluation_report.json`
- `evaluation_report.md`
- `failure_summary.json`
- `failure_summary.md`
- `v0_v1_comparison_summary.json`
- `v0_v1_comparison_summary.md`
- `scene_level_delta_report.json`
- `scene_level_delta_report.md`
- `stratified_v0_v1_summary.json`
- `stratified_v0_v1_summary.md`
- `first_research_result.json`
- `first_research_result.md`
- `trustworthy_comparison_status.json`
- `trustworthy_comparison_status.md`
- `next_improvement_decision.json`
- `next_improvement_decision.md`
- `researcher_summary.json`
- `researcher_summary.md`
- `post_true_v1_analysis.json`
- `external_propagation_summary.json`
- `external_propagation_summary.md`
- `readiness_summary.json`
- `readiness_summary.md`
- `environment_report.json`

Failure summaries include:

- overall failure counts
- per-setting failure counts
- calibration-focused failure counts (`wrong_up_axis`, `manhattan_ambiguity`, `insufficient_plane_evidence`, etc.)

`evaluation_report.json` also includes:

- `v1_execution_summary`:
  - `num_true_v1_execution`
  - `num_fallback_used`
  - `avg_calibration_reliability`
  - `avg_manhattan_ambiguity`
  - `avg_scale_drift`
  - downstream violation counts by setting
- `v0_v1_comparison_summary`:
  - aggregate improvements/regressions
  - scene-level deltas
  - fallback scene list
  - top calibration-related failure categories

Each successful setting also stores additive language-facing metadata under `setting.metadata`:

- `structured_prediction_pre_repair`
- `structured_prediction_post_repair`
- `language_export_pre_repair`
- `language_export_post_repair`

These fields are deterministic exports derived from structured scene predictions and can be used to build JSONL artifacts for language-facing experiments without changing calibration evaluation schema.

Use these to inspect where gains come from across calibration and repair settings.

Post-run analysis adds:

- partitioned summaries for:
  - all scenes
  - true-v1 execution scenes only
  - fallback-used scenes only
  - partial-calibration scenes (if present in execution metadata)
- stratified comparison tables by:
  - `source_type`
  - scene tags
  - fallback reason
  - major failure category
  - true-v1-only and fallback-only slices
- a trustworthiness status block for v0-v1 validity
- a concise first-result summary with one prioritized v1.1 improvement target
- a next-improvement decision artifact with explicit v1.1 evidence checklist
- a researcher-facing compact summary (`researcher_summary.md`)
- propagation-aware scene deltas:
  - pre-repair prediction summary per setting (`prediction_summary_v0/v1/v1_plus_repair`)
  - prediction deltas (`prediction_delta_v1_minus_v0`, etc.)
  - calibrated-input summaries and propagation diagnostics per setting

Interpretation guidance:

- If fallback count is high, do not treat v1-v0 differences as evidence for the true plane-aware method.
- Prioritize scenes with `true_v1_execution=true` when making calibration claims.
- If `num_true_v1_execution == 0`, reports include a trustworthiness warning and the run is not a valid v0-v1 method comparison.
- If `trustworthy_comparison_status.conclusions_provisional == true`, treat all conclusions as provisional.
- The repo selects one v1.1 target from run evidence using fallback ratio + calibration failure taxonomy + partition deltas.
- Start analysis with:
  - `researcher_summary.md`
  - `trustworthy_comparison_status.md`
  - `next_improvement_decision.md`
  - `scene_level_delta_report.md`
  - `external_propagation_summary.md` (external execution status + pre-repair external-vs-mock deltas)
  - `stratified_v0_v1_summary.md`
  - then check `scene_level_delta_report.md` prediction/propagation sections to verify whether calibration changes reached generator inputs and pre-repair predictions.

Language-facing post-processing (run separately, not part of default eval workflow):

```bash
PYTHONPATH=src python scripts/build_language_dataset.py \
  --evaluation-report outputs/eval_pack/latest/evaluation_report.json \
  --output-dir outputs/eval_pack/latest/language
```
