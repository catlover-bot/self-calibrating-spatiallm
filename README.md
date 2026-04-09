# self-calibrating-spatiallm

Pipeline-centric research repository for self-calibrating structured 3D scene understanding.

Core path:

`raw point cloud -> calibration -> structured scene generation -> structural repair -> actionable scene representation -> evaluation`

The repository now supports:

- real point-cloud loading (`.ply`, `.pcd`, `.npy`, `.npz`)
- calibration methods:
  - `geometric_v0` (baseline)
  - `plane_aware_v1` (plane-role + Manhattan-aware)
- mock and external SpatialLM generator modes
- single-scene and small multi-scene runs
- lightweight annotation-driven evaluation packs for evidence-building
- quantitative setting comparisons and failure taxonomy summaries
- post-run v0-v1 analysis artifacts (true-v1 only, fallback-only, scene deltas, stratified summaries)
- v1.5 propagation diagnostics for calibration-to-generation sensitivity checks

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Minimal install for core ASCII `.ply/.pcd` pipeline paths:

```bash
pip install -e .
```

Dependencies for true `calibration_v1` execution and `.npy/.npz` loading:

```bash
pip install -e ".[calibration_v1]"
```

Dependencies for tests:

```bash
pip install -e ".[test]"
```

Point-cloud dependency notes:

- `numpy` is required for:
  - true `plane_aware_v1` execution
  - `.npy/.npz` point-cloud loading
- baseline ASCII `.ply/.pcd` loading works without heavy 3D libs

## Calibration v1

`plane_aware_v1` adds a simple but inspectable geometric upgrade over `geometric_v0`:

- extracts dominant plane candidates from point geometry
- infers floor/ceiling/wall roles when possible
- estimates up-axis and Manhattan-style horizontal orientation with confidence
- applies v1.1 safety guardrails for up-axis selection:
  - penalizes wall-like up candidates via compactness + room-height plausibility
  - optionally applies partial calibration override when v1 up-axis strongly disagrees with v0
- applies v1.2 horizontal safety guardrails:
  - downgrades to partial calibration when Manhattan ambiguity is high
  - uses deterministic up-only horizontal handling when horizontal evidence is weak
- applies v1.3 confidence/reliability calibration:
  - uses dynamic horizontal acceptance thresholds driven by ambiguity
  - distinguishes reliability modes (`full_calibration`, `safe_partial_calibration`, `degraded_fallback`)
  - records reliability component breakdown for interpretability
- applies v1.4 plane-interpretation scoring:
  - deduplicates orientation-equivalent wall candidates
  - scores wall candidates using structural plausibility + PCA alignment
  - records ranked wall candidates and horizontal evidence quality diagnostics
- applies canonical normalization only when reliability is sufficient
- records fallback reason and drops to v0 behavior when plane evidence is weak

`plane_aware_v1` uses `numpy` for plane/covariance analysis. If `numpy` is unavailable, it gracefully falls back to v0 and records `fallback_reason`.

Key diagnostics live in `03_calibration_result.json`:

- `metadata.diagnostics.plane_candidates`
- `metadata.diagnostics.selected_plane_roles`
- `metadata.diagnostics.axis_candidates`
- `metadata.diagnostics.up_axis_selection` (v1.1 scoring rationale)
- `metadata.diagnostics.up_axis_guardrail` (v1.1 safety decision)
- `metadata.diagnostics.horizontal_guardrail` (v1.2 horizontal acceptance decision)
- `metadata.diagnostics.reliability_breakdown` (v1.3 reliability component scoring)
- `metadata.diagnostics.wall_candidate_ranking` (v1.4 ranked wall candidates with scores)
- `metadata.diagnostics.horizontal_evidence` (v1.4 horizontal evidence summary)
- `metadata.diagnostics.confidence`
- `metadata.diagnostics.fallback_reason` (when used)
- `metadata.execution` (`true_v1_execution`, `fallback_used`, `partial_calibration_applied`, `candidate_plane_count`, `weak_scale_reasoning_active`)

## Environment Check

Check if the runtime can execute true v1 calibration, available point-cloud backends, and external SpatialLM adapter support:

```bash
PYTHONPATH=src python scripts/check_environment.py --format text
```

Or via CLI:

```bash
PYTHONPATH=src python -m self_calibrating_spatiallm.cli check-env --format json
```

See `docs/environment.md` for interpretation guidance.

## True-v1 Workflow (Recommended)

1. Install true-v1 + test extras:

```bash
pip install -e ".[calibration_v1,test]"
```

Or bootstrap with helper script:

```bash
bash scripts/setup_true_v1_env.sh
```

2. Run environment check:

```bash
PYTHONPATH=src python scripts/check_environment.py --format text
```

3. Confirm readiness:

- `true_calibration_v1_ready: True`
- `v0_v1_method_comparison_ready: True`

4. Run evaluation pack:

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

- `outputs/eval_pack/latest/researcher_summary.md`
- `outputs/eval_pack/latest/trustworthy_comparison_status.md`
- `outputs/eval_pack/latest/next_improvement_decision.md`
- `outputs/eval_pack/latest/scene_level_delta_report.md`
- `outputs/eval_pack/latest/external_propagation_summary.md` (for external SpatialLM propagation status/deltas)

One-command workflow helper:

```bash
bash scripts/run_true_v1_workflow.sh
```

Makefile equivalents:

```bash
make install-true-v1
make check-env
make eval-v1
make failure-v1
make post-true-v1-analysis
```

## Run Pipeline

Single scene:

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --sample-config configs/samples/real_scene_config.json \
  --calibration-mode v1 \
  --output-dir outputs/runs/single_scene_real
```

Small multi-scene qualitative run:

```bash
PYTHONPATH=src python scripts/run_multi_scene.py \
  --config-list configs/samples/multi_scene_small.json \
  --output-dir outputs/runs/multi_scene_real
```

## External SpatialLM Wiring

Set `generator_mode` to `external` and configure command via:

1. `spatiallm_command` in scene config
2. env var (default): `SCSLM_SPATIALLM_COMMAND`

Example:

```bash
export SCSLM_SPATIALLM_COMMAND='python /path/to/spatiallm_wrapper.py --input {spatiallm_input} --output {output_json} --scene {scene_id}'
PYTHONPATH=src python scripts/run_pipeline.py \
  --sample-config configs/samples/real_scene_external_template.json
```

Supported placeholders:

- `{spatiallm_input}`
- `{input_json}`
- `{output_json}`
- `{scene_id}`

To validate external-path propagation in eval-pack runs, inspect:

- `outputs/eval_pack/latest/external_propagation_summary.md`
- `outputs/eval_pack/latest/external_propagation_summary.json`

If `external_path_executed` is `False`, external propagation claims are not yet validated.

## Lightweight Annotation Format

See example files in `configs/annotations/` and schema:

- `schemas/scene_annotation.schema.json`

Supported fields include:

- `scene_id`
- `expected_up_axis`
- `expected_horizontal_axis` / `room_orientation_hint`
- `expected_scale_hint`
- `expected_door_count`, `expected_window_count`
- `expected_object_categories`
- `expected_relations` (category-level simple relations)
- `traversability_labels`
- `notes`

## Evaluation Pack

Manifest schema:

- `schemas/evaluation_pack_manifest.schema.json`

Example pack:

- `configs/eval_pack/small_eval_pack.json` (5 scenes)

Run evaluation pack:

```bash
PYTHONPATH=src python scripts/run_eval_pack.py \
  --manifest configs/eval_pack/small_eval_pack.json \
  --output-dir outputs/eval_pack/latest
```

Run failure analysis from evaluation report:

```bash
PYTHONPATH=src python scripts/run_failure_analysis.py \
  --evaluation-report outputs/eval_pack/latest/evaluation_report.json \
  --output-dir outputs/eval_pack/latest
```

## Quantitative Comparisons

Evaluation pack compares settings:

- `no_calibration`
- `calibration_v0`
- `calibration_v1`
- `calibration_v1_plus_repair`
- `mock_generator`
- `external_generator` (when configured)

Metrics include:

- calibration quality (`up-axis error`, `horizontal error`, scale match/error)
- calibration reliability (`up-axis confidence`, `horizontal confidence`, `overall reliability`, `manhattan ambiguity`)
- structured scene quality (category presence P/R/F1, door/window count error, pre-repair violations)
- repair effect (violations before/after, fixes applied, overcorrection flag)
- actionable quality (simple relation P/R/F1, traversability accuracy when labels exist)

Evaluation outputs also include v1 execution quality summary:

- number of scenes using true v1 path
- number of fallback cases
- average calibration reliability
- average Manhattan ambiguity
- average scale drift
- downstream violation counts by setting

Post-run analysis outputs include:

- `trustworthy_comparison_status.{json,md}` (explicit valid/provisional status block for v0-v1 claims)
- `scene_level_delta_report.{json,md}` (v0/v1/v1+repair metrics + deltas per scene, execution/fallback diagnostics, pre-repair prediction and propagation deltas)
- `stratified_v0_v1_summary.{json,md}` (grouped by source type, tags, fallback reason, failure category, true-v1-only/fallback-only slices)
- `next_improvement_decision.{json,md}` (single chosen v1.1 target + reason + success evidence checklist)
- `first_research_result.{json,md}` (trustworthiness, key gains/regressions, top failures, prioritized next target)
- `researcher_summary.{json,md}` (compact read-first summary for post-true-v1 interpretation)
- `post_true_v1_analysis.json` (bundle of partition, stratified, decision, and researcher summary outputs)

Dedicated post-true-v1 flow (for already completed eval runs):

```bash
PYTHONPATH=src python scripts/run_post_true_v1_analysis.py \
  --evaluation-report outputs/eval_pack/latest/evaluation_report.json \
  --output-dir outputs/eval_pack/latest
```

## Failure Taxonomy

Failure categories include:

- wrong_up_axis
- unstable_up_axis_confidence
- wrong_horizontal_axis
- manhattan_ambiguity
- scale_drift
- insufficient_plane_evidence
- missing_structural_elements
- implausible_object_sizes
- clutter_dominated_failure
- repair_overcorrection
- relation_derivation_failure
- non_manhattan_scene_failure

## What To Inspect First

For pipeline runs:

1. `run_manifest.json`
2. `03_calibration_result.json` (v1 diagnostics, confidence, fallback reason)
3. `04a_propagation_diagnostics.json` (calibration-to-generation sensitivity summary for mock/external adapters)
4. `04c_generator_execution.json`
5. `10_qualitative_report.md`
6. `08_ablation_report.json`

For evaluation packs:

1. `evaluation_report.json`
2. `researcher_summary.md`
3. `trustworthy_comparison_status.md`
4. `next_improvement_decision.md`
5. `scene_level_delta_report.md`
6. `stratified_v0_v1_summary.md`
7. `first_research_result.md`
8. `failure_summary.md`
9. `v0_v1_comparison_summary.md`

Fallback-heavy interpretation:

- if `num_true_v1_execution == 0`, this is NOT a valid v0-v1 method comparison
- if `num_fallback_used` is close to or exceeds `num_true_v1_execution`, treat v0-v1 conclusions as infrastructure-only
- if `trustworthy_comparison_status.conclusions_provisional == true`, treat conclusions as provisional
- verify `calibration_execution` fields before claiming calibration-v1 improvements

## Repository Layout

```text
.
├── configs/
│   ├── samples/                     # Scene configs
│   ├── annotations/                 # Lightweight per-scene annotations
│   └── eval_pack/                   # Evaluation pack manifests
├── scripts/
│   ├── run_pipeline.py
│   ├── run_multi_scene.py
│   ├── check_environment.py
│   ├── setup_true_v1_env.sh
│   ├── run_true_v1_workflow.sh
│   ├── run_eval_pack.py
│   ├── run_failure_analysis.py
│   └── run_post_true_v1_analysis.py
├── schemas/                         # JSON schemas for configs/artifacts
├── src/self_calibrating_spatiallm/
│   ├── io/
│   ├── calibration/
│   ├── generation/
│   ├── repair/
│   ├── scene_graph/
│   ├── evaluation/
│   ├── visualization/
│   └── pipeline/
└── tests/
```
