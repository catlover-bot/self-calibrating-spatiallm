# ARKitScenes Local Scaffold

This directory holds local ARKitScenes manifest-generation templates.

Start from:
- `dataset_config.template.json`

Copy it to a local file (for example `dataset_config.local.json`) and set:
- `dataset_root`

Then run:

```bash
PYTHONPATH=src python scripts/build_arkitscenes_manifest.py \
  --dataset-config configs/public_datasets/arkitscenes/dataset_config.local.json
```

Generated files are written under `configs/public_datasets/arkitscenes/generated/` by default.
