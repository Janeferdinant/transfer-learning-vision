# Common tasks. Long experiments are never run by CI; see README for expected times.
PY ?= python
SEEDS5 = 0 1 2 3 4
SEEDS3 = 0 1 2

.PHONY: install lint typecheck test test-full check pilot backbones data-efficiency ablations gradcam analyse explorer all-experiments

install:
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check .

typecheck:
	mypy

test:            ## fast tests (no TensorFlow needed; TensorFlow tests are skipped if it is missing)
	$(PY) -m pytest -q

test-full: test  ## same, plus model-construction tests (needs TensorFlow, no weight download)
	$(PY) -m pytest -q tests/test_models.py

check: lint typecheck test

# ---- experiments (CPU, sequential; wall-clock times are measured, do not run other heavy jobs) ----
pilot:
	$(PY) -m tlv.suite --suite pilot --dataset flowers --seeds 100 --out results/pilot_flowers
	$(PY) -m tlv.suite --suite pilot --dataset eurosat --seeds 100 --out results/pilot_eurosat

backbones:
	$(PY) -m tlv.suite --suite backbones --dataset flowers --seeds $(SEEDS5) --out results/backbones_flowers
	$(PY) -m tlv.suite --suite backbones --dataset eurosat --seeds $(SEEDS5) --out results/backbones_eurosat

data-efficiency:
	$(PY) -m tlv.suite --suite data-efficiency --dataset flowers --seeds $(SEEDS5) --out results/data_efficiency_flowers
	$(PY) -m tlv.suite --suite data-efficiency --dataset eurosat --seeds $(SEEDS5) --out results/data_efficiency_eurosat

ablations:
	$(PY) -m tlv.suite --suite depth --dataset flowers --seeds $(SEEDS3) --out results/ablation_depth_flowers
	$(PY) -m tlv.suite --suite depth --dataset eurosat --seeds $(SEEDS3) --out results/ablation_depth_eurosat
	$(PY) -m tlv.suite --suite lr --dataset flowers --seeds $(SEEDS3) --out results/ablation_lr_flowers
	$(PY) -m tlv.suite --suite image-size --dataset flowers --seeds $(SEEDS3) --out results/ablation_size_flowers

gradcam:
	$(PY) -m tlv.gradcam --out results/gradcam

analyse:
	$(PY) -m tlv.analysis
	$(PY) -m tlv.plots

explorer:
	$(PY) -m tlv.export_viz --date $(DATE)

all-experiments: pilot backbones data-efficiency ablations gradcam analyse
