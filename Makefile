PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTHONPATH ?= src

.PHONY: setup setup-true-v1 install-true-v1 lint format typecheck test check pipeline pipeline-multi eval-pack failure-analysis check-env eval-v1 failure-v1 debug-v1 v1-workflow post-true-v1-analysis

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
