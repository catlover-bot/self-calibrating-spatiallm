# Robustness-Boundary Workflow

This workflow studies calibration decision boundaries under controlled input perturbations.

Core question:
- Under what input conditions does calibration remain helpful?
- Where does reliability begin to degrade?
- How do those changes propagate into structured outputs and language-facing artifacts?

The workflow is additive. It does not replace or alter the existing eval-pack schema.

## Entry Points

- Config: `configs/robustness_boundary.default.json`
- Runner: `scripts/run_robustness_boundary.py`
- Make target: `make robustness-boundary`

## Perturbation Families

Default perturbations:
- `rotation_yaw`
- `tilt_up_axis`
- `manhattan_degradation`
- `structural_dropout`
- `clutter_injection`
- `density_sparsity`

Each perturbation supports multiple severity values and deterministic seeds.

Each generated variant records:
- `base_scene_id`
- `perturbation_type`
- `severity`
- `severity_bucket`
- `seed`
- exact perturbation parameters

## Split Framing

The default config makes split roles explicit:
- `small`: `clean_validation`
- `public`: `noisy_realism`

This supports analysis that separates clean-signal validation from real-world high-ambiguity regimes.

## Commands

```bash
make robustness-boundary
```

Equivalent direct command:

```bash
PYTHONPATH=src python scripts/run_robustness_boundary.py \
  --config configs/robustness_boundary.default.json \
  --output-dir outputs/eval_pack/robustness_boundary/latest
```

## Output Layout

Root outputs:
- `robustness_boundary_config.resolved.json`
- `robustness_boundary_run_manifest.json`
- `perturbation_inventory.json`
- `robustness_boundary_rows.jsonl`
- `robustness_boundary_summary.json`
- `robustness_boundary_summary.md`

Per-split eval outputs:
- `outputs/eval_pack/robustness_boundary/latest/splits/<split_name>/evaluation_report.json`
- standard eval-pack reports under each split directory

Language-facing boundary outputs:
- `language/robustness_language_scene_examples.jsonl`
- `language/robustness_language_qa_examples.jsonl`
- `language/robustness_language_grounding_examples.jsonl`
- `language/robustness_language_severity_deltas.jsonl`
- `language/robustness_language_summary.json`
- `language/robustness_language_summary.md`

## Boundary Analysis Signals

The summary includes:
- grouping by split + perturbation + severity
- comparison-valid and partial-calibration rates
- ambiguity and confidence trends
- v1-v0 metric deltas:
  - `calibration_reliability`
  - `calibration_horizontal_error_deg`
  - `calibration_up_axis_error_deg`
  - `structured_violation_count_before_repair`
  - `repair_violations_after`
  - `actionable_relation_f1`
- external-vs-mock divergence aggregates when available
- major failure category and fallback reason summaries
- heuristic boundary findings (help-until / degradation-onset style statements)

## Interpretation Tips

Inspect in this order:
1. `robustness_boundary_summary.md`
2. `robustness_boundary_summary.json`
3. `robustness_boundary_rows.jsonl`
4. `language/robustness_language_summary.md`
5. `language/robustness_language_severity_deltas.jsonl`

Use `robustness_boundary_rows.jsonl` for scene-level drill-down by:
- perturbation family
- severity
- base scene
- partial calibration
- failure category
- external-vs-mock prediction deltas

