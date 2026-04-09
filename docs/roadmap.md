# Roadmap

## Phase 0: Real Single-Scene + External Adapter (completed)

- Real point cloud loading (.ply/.pcd/.npy/.npz)
- Geometric calibration baseline with diagnostics
- External SpatialLM command adapter with diagnostics capture
- End-to-end saved artifacts, run manifest, qualitative markdown report
- Single-scene structural + generator comparison outputs

## Phase 1: Evidence-Building + Calibration Method Upgrade (current)

- Lightweight per-scene annotations
- 5-scene evaluation pack manifest
- Quantitative per-setting metrics
- Aggregate comparison tables
- Failure taxonomy summaries
- Plane-aware calibration_v1 with fallback diagnostics
- Explicit calibration_v0 vs calibration_v1 comparison settings

## Phase 2: External Model Integration Hardening

- Harden external SpatialLM wrapper contracts
- Add richer parse adapters for multiple output layouts
- Expand generator-level diagnostics and error taxonomies

## Phase 3: Repair + Structure Research

- Expand rule set and constraint handling
- Add confidence-aware repair decision logic
- Add structural consistency metrics

## Phase 4: Evaluation Rigor

- Add task-specific metrics
- Add reproducible multi-scene runner with fixed splits
- Add automated report aggregation and failure clustering
