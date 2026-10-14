#!/usr/bin/env bash
# Sequential experiment queue used for the 2026-10 additions (see docs/runs/). Resumable.
set -u
PY="${PY:-python}"
run() { echo "=== $(date -u +%FT%TZ) $*"; "$PY" -m tlv.suite "$@" 2>&1 | grep -v -E "^I0000|^W0000|UserWarning|get_data_adapter|oneDNN"; }
run --suite pilot --dataset flowers --seeds 100 --out results/pilot_flowers
run --suite pilot --dataset eurosat --seeds 100 --out results/pilot_eurosat
run --suite backbones --dataset flowers --seeds 0 1 2 3 4 --out results/backbones_flowers
run --suite backbones --dataset eurosat --seeds 0 1 2 3 4 --out results/backbones_eurosat
run --suite data-efficiency --dataset flowers --seeds 0 1 2 3 4 --out results/data_efficiency_flowers
run --suite data-efficiency --dataset eurosat --seeds 0 1 2 3 4 --out results/data_efficiency_eurosat
run --suite depth --dataset flowers --seeds 0 1 2 --out results/ablation_depth_flowers
run --suite lr --dataset flowers --seeds 0 1 2 --out results/ablation_lr_flowers
run --suite image-size --dataset flowers --seeds 0 1 2 --out results/ablation_size_flowers
run --suite depth --dataset eurosat --seeds 0 1 2 --out results/ablation_depth_eurosat
echo "=== $(date -u +%FT%TZ) queue finished"
