#!/usr/bin/env bash
# Sequential experiment queue used for the 2026-10 additions (see docs/runs/). Resumable: a suite
# that is interrupted (for example by a memory shortage on a shared desktop) is retried and
# continues from the runs already in its per_run.csv.
set -u
set -o pipefail
PY="${PY:-python}"
run() {
  for attempt in 1 2 3 4; do
    echo "=== $(date -u +%FT%TZ) attempt $attempt: $*"
    "$PY" -m tlv.suite "$@" 2>&1 | grep --line-buffered -v -E "^I0000|^W0000|UserWarning|get_data_adapter|oneDNN|retracing|return MobileNetV3"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then return 0; fi
    sleep 30
  done
}
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
