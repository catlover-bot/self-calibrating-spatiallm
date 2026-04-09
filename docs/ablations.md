# Single-Scene Comparisons

Each single-scene run emits two comparison groups.

## Structural Settings

1. `no_calibration`
2. `calibration_v0`
3. `calibration_v1`
4. `calibration_v1_plus_repair`

These compare how calibration and repair stages affect downstream structure/action consistency.

## Generator Settings

- `mock_generator`
- `external_generator` (when configured)

These compare generator paths while keeping calibration/repair fixed to plane-aware-v1 + rule-based defaults.

## Purpose

This is a qualitative diagnostics layer, not a benchmark suite.
It helps answer:

- whether calibration is helping scene/action consistency on a given scene
- whether repair is correcting obvious structural errors
- whether external generator output quality/format differs from mock assumptions

## Output Location

Comparison results are saved in:

- `08_ablation_report.json`
- `09_summary.txt`
- `10_qualitative_report.md`
