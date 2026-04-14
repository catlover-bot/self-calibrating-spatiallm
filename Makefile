PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTHONPATH ?= src
ARKITSCENES_ROOT ?= /absolute/path/to/ARKitScenes
ARKITSCENES_SUBSET_SIZE ?= 25
ARKITSCENES_SEED ?= 13
PUBLIC_MANIFEST ?= configs/public_datasets/arkitscenes/generated/arkitscenes_subset_manifest.json
PUBLIC_EVAL_OUTPUT ?= outputs/eval_pack/public_medium_latest
PUBLIC_BUILD_ARGS ?=
LANG_EVAL_REPORT ?= outputs/eval_pack/latest/evaluation_report.json
LANG_OUTPUT_DIR ?= outputs/eval_pack/latest/language
SCENE_PREDICTION ?= outputs/runs/single_scene_real/04_scene_prediction.json
DEMO_HOST ?= 127.0.0.1
DEMO_PORT ?= 8765
DEMO_PRESET ?= small_best_demo
DEMO_STEP_SECONDS ?= 9

.PHONY: setup setup-true-v1 install-true-v1 lint format typecheck test check pipeline pipeline-multi eval-pack failure-analysis check-env eval-v1 failure-v1 debug-v1 v1-workflow post-true-v1-analysis build-arkitscenes-manifest validate-public-manifest public-workflow build-language-dataset export-scene-language demo demo-autoplay

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

setup-true-v1:
	bash scripts/setup_true_v1_env.sh

install-true-v1:
	$(PIP) install -e ".[calibration_v1,test]"

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src

test:
	pytest

check: lint typecheck test

pipeline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py --sample-config configs/samples/real_scene_config.json --output-dir outputs/runs/single_scene_real

pipeline-multi:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_multi_scene.py --config-list configs/samples/multi_scene_small.json --output-dir outputs/runs/multi_scene_real

eval-pack:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval_pack.py --manifest configs/eval_pack/small_eval_pack.json --output-dir outputs/eval_pack/latest

failure-analysis:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_failure_analysis.py --evaluation-report outputs/eval_pack/latest/evaluation_report.json --output-dir outputs/eval_pack/latest

check-env:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/check_environment.py --format text

eval-v1:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval_pack.py --manifest configs/eval_pack/small_eval_pack.json --output-dir outputs/eval_pack/latest

failure-v1:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_failure_analysis.py --evaluation-report outputs/eval_pack/latest/evaluation_report.json --output-dir outputs/eval_pack/latest

debug-v1:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_pipeline.py --sample-config configs/samples/real_scene_config.json --calibration-mode v1 --output-dir outputs/runs/single_scene_real

v1-workflow:
	bash scripts/run_true_v1_workflow.sh

post-true-v1-analysis:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_post_true_v1_analysis.py --evaluation-report outputs/eval_pack/latest/evaluation_report.json --output-dir outputs/eval_pack/latest

build-arkitscenes-manifest:
	PYTHONPATH=src $(PYTHON) scripts/build_arkitscenes_manifest.py --dataset-root $(ARKITSCENES_ROOT) --subset-size $(ARKITSCENES_SUBSET_SIZE) --seed $(ARKITSCENES_SEED) $(PUBLIC_BUILD_ARGS)

validate-public-manifest:
	PYTHONPATH=src $(PYTHON) scripts/validate_public_dataset_manifest.py --manifest $(PUBLIC_MANIFEST) --check-load --format text --summary-output $(PUBLIC_EVAL_OUTPUT)/public_manifest_validation.json

public-workflow:
	MANIFEST_PATH=$(PUBLIC_MANIFEST) EVAL_OUTPUT_DIR=$(PUBLIC_EVAL_OUTPUT) bash scripts/run_public_dataset_workflow.sh

build-language-dataset:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_language_dataset.py --evaluation-report $(LANG_EVAL_REPORT) --output-dir $(LANG_OUTPUT_DIR)

export-scene-language:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/export_scene_prediction_language.py --scene-prediction $(SCENE_PREDICTION)

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/serve_demo.py --host $(DEMO_HOST) --port $(DEMO_PORT) --open-browser

demo-autoplay:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/serve_demo.py --host $(DEMO_HOST) --port $(DEMO_PORT) --open-browser --autoplay --preset $(DEMO_PRESET) --step-seconds $(DEMO_STEP_SECONDS) --loop
