# Architecture Notes

The repository is organized around modular stage interfaces:

- `Calibrator`
- `SceneGenerator`
- `SceneRepairer`
- `ActionableSceneBuilder`

These interfaces are composed in `pipeline.SingleScenePipeline`.

Current calibration implementations:

- `GeometricCalibratorV0` (simple covariance/range baseline)
- `PlaneAwareCalibratorV1` (dominant plane role inference + confidence-gated normalization with fallback)

## Why This Matters

- Stage swap for ablations without rewriting orchestration
- Stable artifact schemas for debugging and tooling
- Clear separation between operational pieces and placeholder model components

## External SpatialLM Integration

`generation.SpatialLMExternalGenerator` isolates command invocation, raw output capture, and parsing into `ScenePrediction`, so model integration can evolve without changing downstream stages.

## Inspectability Contract

Pipeline runs prioritize traceability:

- explicit stage timestamps/status in `run_manifest.json`
- stable intermediate artifact files
- markdown qualitative report per scene
- small multi-scene runner for quick failure clustering
- explicit environment capability checks (`check-env`) for true-v1 vs fallback interpretation

## Evaluation Layer

Evidence-building modules in `evaluation/` add:

- lightweight annotation loaders
- small evaluation pack manifest support
- per-setting quantitative metrics
- aggregate comparison tables
- failure taxonomy classification
