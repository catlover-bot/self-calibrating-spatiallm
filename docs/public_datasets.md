# Public Dataset Experiments

This project is now ready for public-dataset robustness experiments.

Primary target: **ARKitScenes**

Why ARKitScenes first:
- mobile RGB-D / LiDAR indoor capture is aligned with the research direction
- noisy, real-world scene capture stresses calibration and propagation quality
- better match to casual/mobile capture than curated static scans

## Scope

Use **medium subsets (20-30 scenes)** first.
Do not jump directly to full-dataset benchmarking.

The goal at this stage is controlled robustness analysis of:
- calibration quality and confidence behavior
- calibration-to-generation propagation
- pre-repair structured scene fidelity
- repair dependence

## Local ARKitScenes Setup

1. Prepare a local dataset root (any internal directory layout is acceptable).

The scaffold script recursively searches for point clouds by extension, so exact folder shape can vary. A typical local tree might look like:

```text
/your/path/ARKitScenes/
  Training/
    <scene_a>/
      *.ply
    <scene_b>/
      *.ply
  Validation/
    <scene_c>/
      *.ply
```

2. Copy and edit template:

- `configs/public_datasets/arkitscenes/dataset_config.template.json`

Set at least:
- `dataset_root`

## Build Subset Manifest (Small / Medium)

Via Makefile (recommended):

```bash
make build-arkitscenes-manifest ARKITSCENES_ROOT=/absolute/path/to/ARKitScenes ARKITSCENES_SUBSET_SIZE=10
make build-arkitscenes-manifest ARKITSCENES_ROOT=/absolute/path/to/ARKitScenes ARKITSCENES_SUBSET_SIZE=25
```

Small subset (10 scenes):

```bash
PYTHONPATH=src python scripts/build_arkitscenes_manifest.py \
  --dataset-root /absolute/path/to/ARKitScenes \
  --subset-size 10 \
  --seed 13
```

Medium subset (25 scenes, recommended first public run):

```bash
PYTHONPATH=src python scripts/build_arkitscenes_manifest.py \
  --dataset-root /absolute/path/to/ARKitScenes \
  --subset-size 25 \
  --seed 13
```

Generated artifacts (default):
- `configs/public_datasets/arkitscenes/generated/arkitscenes_subset_manifest.json`
- `configs/public_datasets/arkitscenes/generated/arkitscenes_scene_inventory.json`
- `configs/public_datasets/arkitscenes/generated/sample_configs/*.json`

## Validate Subset Usability

Via Makefile:

```bash
make validate-public-manifest
```

```bash
PYTHONPATH=src python scripts/validate_public_dataset_manifest.py \
  --manifest configs/public_datasets/arkitscenes/generated/arkitscenes_subset_manifest.json \
  --check-load \
  --format text
```

Optional JSON summary:

```bash
PYTHONPATH=src python scripts/validate_public_dataset_manifest.py \
  --manifest configs/public_datasets/arkitscenes/generated/arkitscenes_subset_manifest.json \
  --check-load \
  --format json \
  --summary-output outputs/eval_pack/public_medium_latest/public_manifest_validation.json
```

## Run Public-Dataset Workflow

Via Makefile:

```bash
make public-workflow
```

```bash
MANIFEST_PATH=configs/public_datasets/arkitscenes/generated/arkitscenes_subset_manifest.json \
EVAL_OUTPUT_DIR=outputs/eval_pack/public_medium_latest \
bash scripts/run_public_dataset_workflow.sh
```

Equivalent manual sequence:
- `scripts/run_eval_pack.py`
- `scripts/run_failure_analysis.py`
- `scripts/run_post_true_v1_analysis.py`

## Artifacts To Inspect First

- `outputs/eval_pack/public_medium_latest/researcher_summary.md`
- `outputs/eval_pack/public_medium_latest/trustworthy_comparison_status.md`
- `outputs/eval_pack/public_medium_latest/scene_level_delta_report.md`
- `outputs/eval_pack/public_medium_latest/stratified_v0_v1_summary.md`
- `outputs/eval_pack/public_medium_latest/external_propagation_summary.md`
- `outputs/eval_pack/public_medium_latest/evaluation_report.md`

## What To Check First

1. Trustworthiness and v1 execution status (`fallback` / `partial` rates)
2. v0->v1 calibration deltas (horizontal, up-axis, reliability)
3. pre-repair violation trends
4. external-vs-mock pre-repair prediction deltas
5. whether gains are calibration-driven or repair-dominated

## External Quality Caution

External SpatialLM is scene-dependent.
Even when external execution is successful, semantic fidelity can vary across scenes.
Treat medium-subset runs as robustness diagnostics before any large-scale claim.

## Optional Next Targets (Scaffold Only)

Templates are provided for future extension:
- `configs/public_datasets/3rscan/dataset_config.template.json`
- `configs/public_datasets/3dssg/dataset_config.template.json`

Current first-class implementation is ARKitScenes.
