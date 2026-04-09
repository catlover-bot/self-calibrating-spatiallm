# SpatialLM Integration Guide

This repository expects SpatialLM to be connected as an external command adapter.

## Command Wiring

The external adapter (`generation.external.SpatialLMExternalGenerator`) resolves command in this order:

1. `spatiallm_command` in scene config
2. environment variable from `spatiallm_command_env_var` (default: `SCSLM_SPATIALLM_COMMAND`)

Supported placeholders:

- `{spatiallm_input}`
- `{input_json}`
- `{output_json}`
- `{scene_id}`

## Expected Wrapper Behavior

Your local wrapper should:

1. read exported `spatiallm_input_v1` payload from `{spatiallm_input}`
2. run model inference
3. emit structured prediction JSON:
   - either write to `{output_json}`
   - or print JSON object to stdout

## Expected Output Shapes

Preferred full schema:

- `sample_id`
- `generator_name`
- `objects` with `object_id`, `label`, `position`, `size`
- `relations` with `subject_id`, `predicate`, `object_id`

Partial schema is accepted:

- `objects` / `instances`
- `relations` with loose field names (`subject`, `relation`, `object`)

The adapter will attempt partial parsing and preserve warnings in metadata.

## Dependencies

Baseline repository dependencies are lightweight (`numpy`, `pytest`, etc.).
External SpatialLM runtime dependencies are intentionally not pinned in this repo; they are expected to be managed in your local model checkout/wrapper environment.

## Failure Diagnostics

Inspect these first:

1. `04c_generator_execution.json`
2. `04d_generator_stdout.txt`
3. `04e_generator_stderr.txt`
4. `run_manifest.json`

Common failures:

- missing command binary/path
- command timeout
- no parseable JSON output
- partial parse due schema mismatch

On generator-stage failure, the pipeline still writes a failed `run_manifest.json` so stage status and error context remain inspectable.

For eval-pack runs, inspect:

- `external_propagation_summary.md` / `.json`

These summarize whether external execution actually happened, per-scene external status, and detectable pre-repair external-vs-mock prediction deltas.
