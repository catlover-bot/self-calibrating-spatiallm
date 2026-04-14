# Demo Recording Guide

This repository includes a local, artifact-driven research demo UI at `demo/index.html`.

The demo does not run inference. It visualizes existing outputs from:

- `outputs/eval_pack/latest`
- `outputs/eval_pack/public_medium_latest`

## Why This Demo

The UI is designed for short research videos that emphasize:

- setting-to-setting structural changes (`no_calibration`, `v0`, `v1`, `v1_plus_repair`)
- mock vs external generator behavior
- object/relation differences and language-facing summaries
- provenance/evidence level visibility

## Quick Start

Serve the repository root so the demo can load local artifacts:

```bash
PYTHONPATH=src python scripts/serve_demo.py --open-browser
```

Then open:

- `http://127.0.0.1:8765/demo/index.html`

## Autoplay / Presentation Mode

Start in autoplay mode from CLI:

```bash
PYTHONPATH=src python scripts/serve_demo.py \
  --open-browser \
  --autoplay \
  --preset small_best_demo \
  --step-seconds 9 \
  --loop
```

Or directly by URL:

- `http://127.0.0.1:8765/demo/index.html?preset=small_best_demo&autoplay=1&stepSec=9&loop=1`

Keyboard controls:

- `Space`: start/stop autoplay
- `Right Arrow`: next step

## Included Presets

`demo/presets.json` includes:

- `small_best_demo`
- `public_relation_demo`
- `mock_vs_external_demo`

Each preset defines scene transitions, setting pairs, and talking points.

## Recommended First Recording Sequence

1. Run `small_best_demo` in autoplay.
2. Pause on `calibration_v0` vs `calibration_v1` to explain relation changes.
3. Show `calibration_v1` vs `calibration_v1_plus_repair` to separate calibration and repair effects.
4. Show `mock_generator` vs `external_generator` for propagation/backend behavior differences.
5. End on the provenance panel to clarify evidence/source class.

## Notes

- If a split is missing `evaluation_report.json`, the UI marks it unavailable.
- If language JSONL artifacts are missing, the UI falls back to evaluation summaries and labels the source class accordingly.
- Public split is optional; demo remains usable with small split only.
