# Pipeline Overview

Core single-scene path:

`PointCloud file -> PointCloudSample/PointCloudMetadata -> CalibratedPointCloud -> ScenePrediction -> RepairResult -> ActionableScene -> EvaluationResult`

## Stage Details

1. Loading (`io.point_cloud`)
- Reads `.ply`, `.pcd`, `.npy`, `.npz`
- Produces `PointCloudSample` and `PointCloudMetadata`
- Saves raw input metadata as first-class artifact

2. Calibration (`calibration.geometric_v0`, `calibration.plane_aware_v1`)
- `geometric_v0`: covariance/range baseline
- `plane_aware_v1`: dominant-plane candidates, floor/ceiling/wall role inference, Manhattan-aware horizontal estimation
- `plane_aware_v1` confidence gates normalization and falls back to v0 (full or partial) when plane evidence is weak
- if `numpy` is unavailable, `plane_aware_v1` records fallback diagnostics and executes v0 behavior
- Exports rich diagnostics/confidence and transformed cloud metadata

Calibration v1 execution metadata fields:

- `metadata.execution.plane_aware_logic_ran`
- `metadata.execution.true_v1_execution`
- `metadata.execution.fallback_used`
- `metadata.execution.fallback_reason`
- `metadata.execution.candidate_plane_count`
- `metadata.execution.weak_scale_reasoning_active`

3. Generation (`generation`)
- `MockSpatialLMGenerator`: deterministic placeholder
- `SpatialLMExternalGenerator`: external command invocation path
- Exports `spatiallm_input_v1` payload for external adapters

4. Repair (`repair.simple_rule_repairer`)
- Applies lightweight structural consistency fixes
- Preserves issue list and fixes applied

5. Actionable Scene (`scene_graph`)
- Derives relation set (`near`, `supported-by`, `inside`, `intersects`, `accessible`, `facing`)
- Emits actionable directives

6. Evaluation (`evaluation.simple_evaluator`)
- Produces lightweight consistency metrics
- Used for per-setting qualitative comparisons

## SpatialLM Input Contract

The external adapter converts internal calibrated representation to `spatiallm_input_v1` JSON:

- `axis_convention`
- `scale_assumptions`
- `normalization`
- `points_xyz`

Current frame assumptions:

- right-handed coordinate system
- `+Z` is up in canonical calibrated frame
- `+X/+Y` are horizontal axes after calibration

## Artifact Stages For External Integration

When external generation is used, three inspectable stages are saved:

1. raw point cloud metadata (`00_point_cloud_metadata.json`)
2. calibrated point cloud metadata (`02b_calibrated_point_cloud_metadata.json`)
3. exported SpatialLM point cloud payload (`02c_spatiallm_export_point_cloud.json`)

## Manifest + Reports

Each run saves:

- `run_manifest.json`: stage-level timestamps, status, modes, artifact paths
- `run_manifest.json` also includes `calibration_execution` for true-v1 vs fallback distinction
- `10_qualitative_report.md`: concise qualitative scene summary
- `manifest.json`: flat artifact path index

## Environment Verification

Use:

`PYTHONPATH=src python scripts/check_environment.py --format text`

to verify whether true `calibration_v1` execution is available (`numpy` present) and whether external SpatialLM adapter execution is configured.

## Multi-Scene Mode

`scripts/run_multi_scene.py` runs a small list of scene configs and saves:

- `multi_scene_manifest.json`
- `multi_scene_report.md`

This mode is for qualitative inspection and failure analysis, not benchmark-scale evaluation.
