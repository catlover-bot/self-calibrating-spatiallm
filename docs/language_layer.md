# Language-Facing Layer

This repository studies calibration decision under uncertainty for structured 3D world modeling.

This layer makes that contribution NLP-usable without changing the calibration experiment schema:

- deterministic scene-to-text exports from structured scene predictions
- deterministic scene-grounded QA and grounding examples
- JSONL dataset artifacts aligned by scene and setting
- explicit weak/strong evidence markers when only summary-level structure is available

The goal is not to claim language-model performance directly.
The goal is to expose calibration effects in language-usable representations.

Source selection is explicit and inspectable:

1. `explicit_structured_prediction`
2. `structured_prediction_with_hint_only`
3. `summary_reconstructed`

The language builder now reads this contract from evaluation metadata:

- `prediction_source_contract`
- `prediction_artifact_paths`
- `prediction_source_contract_version`

## Why This Matters

The core question remains:

How should calibration decide between strong commitment, partial calibration, or conservative behavior under ambiguity, and how do those decisions propagate downstream?

With this layer, that question can be analyzed through language-facing outputs:

- description quality shifts between `calibration_v0`, `calibration_v1`, and `calibration_v1_plus_repair`
- relation verbalization consistency under `mock_generator` vs `external_generator`
- public-dataset robustness for language-facing artifacts

## New Components

- `src/self_calibrating_spatiallm/language/exports.py`
  - scene summary text
  - object list text
  - relation statements text
  - paragraph-style deterministic text
  - hinted-relation preservation when explicit relation tuples are missing
  - reconstruction transparency for summary-derived objects

- `src/self_calibrating_spatiallm/language/tasks.py`
  - deterministic QA examples
  - deterministic grounding/instruction examples
  - scene-setting language record builder
  - weak grounding statements for non-geometric reconstructed objects
  - relation-evidence QA and relation-evidence grounding statements

- `scripts/build_language_dataset.py`
  - builds language JSONL files from `evaluation_report.json`
  - supports aligned per-scene comparison across settings
  - exports pairwise setting-difference summaries and deterministic comparison QA
  - attempts relation/object recovery from upstream metadata artifacts before summary fallback
  - loads structured prediction artifacts via `prediction_artifact_paths` when inline payloads are absent

- `scripts/export_scene_prediction_language.py`
  - exports one `ScenePrediction` artifact into language-facing forms

## Recommended Workflow (Run Later)

After generating evaluation outputs:

```bash
make build-language-dataset \
  LANG_EVAL_REPORT=outputs/eval_pack/latest/evaluation_report.json \
  LANG_OUTPUT_DIR=outputs/eval_pack/latest/language
```

For one scene prediction artifact:

```bash
make export-scene-language SCENE_PREDICTION=outputs/runs/single_scene_real/04_scene_prediction.json
```

## Output Artifacts

Language dataset builder writes:

- `language_scene_examples.jsonl`
- `language_qa_examples.jsonl`
- `language_grounding_examples.jsonl`
- `language_alignment_examples.jsonl`
- `language_export_summary.json`
- `language_export_summary.md`

Inspect first:

1. `language_export_summary.md`
2. `language_alignment_examples.jsonl`
3. `language_scene_examples.jsonl`
4. `language_qa_examples.jsonl` (check relation-evidence and setting-delta style questions)
5. `language_grounding_examples.jsonl` (check weak vs strong grounding separation)

## Notes

- Exports are deterministic and template-based.
- No black-box language generation is used.
- Existing calibration metrics/evaluation schema are unchanged.
- When only summary-level prediction information is available:
  - relation evidence is preserved as hinted evidence (not silently dropped)
  - grounding is marked weak/approximate rather than pretending geometric precision
- This layer is additive and compatible with:
  - calibration comparisons (`no_calibration`, `v0`, `v1`, `v1_plus_repair`)
  - generator comparisons (`mock`, `external`)
  - small and public dataset packs.
