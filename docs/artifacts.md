# Artifact Definitions

## Core Artifacts

- `PointCloudMetadata`: load-time summary of source and geometry stats
- `PointCloudSample`: in-memory point cloud sample representation
- `CalibratedPointCloud`: points in calibrated frame with calibration context
- `CalibrationResult`: axis/origin estimates and diagnostics
- `ScenePrediction`: generated structured scene objects/relations
- `RepairResult`: violations + applied fixes + repaired scene
- `ActionableScene`: derived relation graph and action directives
- `EvaluationResult`: lightweight metrics and pass/fail
- `AblationReport`: structural + generator setting comparisons

## External SpatialLM Diagnostics Artifacts

- `02b_calibrated_point_cloud_metadata.json`: calibrated frame, normalization, scale assumptions
- `02c_spatiallm_export_point_cloud.json`: exported `spatiallm_input_v1` payload (when external path runs)
- `04c_generator_execution.json`: command, stdout/stderr, return code, parser warnings
- `04d_generator_stdout.txt` and `04e_generator_stderr.txt`: raw process streams when available
- `04f_generator_output_raw.json`: raw external JSON output when available

## Calibration v1 Diagnostics (in `03_calibration_result.json`)

When `plane_aware_v1` is active, `CalibrationResult.metadata.diagnostics` also includes:

- `plane_candidates`: dominant candidate plane summaries
- `selected_plane_roles`: inferred floor/ceiling/wall selections
- `axis_candidates`: intermediate up-axis and horizontal candidates
- `confidence`: `up_axis`, `horizontal_orientation`, `overall_reliability`
- `manhattan_ambiguity`
- `scale_reasoning`
- `fallback_reason` when fallback/partial fallback occurs

## Run-Level Reporting Artifacts

- `run_manifest.json`: stage statuses, timestamps, pipeline modes, artifact map
- `manifest.json`: flat artifact path index
- `09_summary.txt`: concise text summary
- `10_qualitative_report.md`: markdown qualitative report
- `v0_v1_comparison_summary.md` (evaluation pack runs): concise v0-v1 improvement/regression summary
