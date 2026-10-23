#!/usr/bin/env bash
# Timed suites rerun on a quieter machine (2026-10-17). Same suites, seeds, epochs and learning rates as
# scripts/run_queue.sh (which produced the committed results/), written to results/idle_<date>/ so the
# earlier CSVs stay untouched. The pilot is not rerun: it is accuracy-only and its learning-rate
# choices (configs/lr_choices.json) are reused unchanged.
#
# Before every batch of runs a 60 s total-CPU sample is taken (scripts/idle_gate.ps1: mean above 15 %
# means "wait 5 min and re-check, for up to 30 min, then run anyway and record the load"); the sample is
# stored next to the console log. One suite at a time, nothing else runs during a suite.
# Resumable: a run is only written to per_run.csv when it finished, so a cut-off run is repeated.
set -u
set -o pipefail
PY="${PY:-python}"
DATE="${DATE:-2026-10-17}"
OUT="results/idle_${DATE}"
LOGS="docs/runs/idle-${DATE}"
MAXW="${MAXW:-30}"
mkdir -p "$OUT" "$LOGS"

batch() {
  local name="$1"; shift
  echo "=== $(date +%FT%T) batch $name: idle gate"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/idle_gate.ps1 -Log "$LOGS/${name}_idle_check.txt" -MaxWaitMin "$MAXW"
  for attempt in 1 2 3 4; do
    echo "=== $(date +%FT%T) batch $name attempt $attempt: tlv.suite $*"
    "$PY" -m tlv.suite "$@" --out "$OUT/$name" 2>&1 \
      | grep --line-buffered -v -E "^I0000|^W0000|UserWarning|get_data_adapter|oneDNN|retracing|return MobileNetV3" \
      | tee -a "$LOGS/${name}_console.txt"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then return 0; fi
    echo "attempt $attempt failed or was cut off; repeating the unfinished runs"
    sleep 30
  done
}

batch backbones_flowers --suite backbones --dataset flowers --seeds 0 1 2 3 4
batch backbones_eurosat --suite backbones --dataset eurosat --seeds 0 1 2 3 4
batch data_efficiency_flowers --suite data-efficiency --dataset flowers --seeds 0 1 2 3 4
batch data_efficiency_eurosat --suite data-efficiency --dataset eurosat --seeds 0 1 2 3 4
batch ablation_depth_flowers --suite depth --dataset flowers --seeds 0 1 2
batch ablation_lr_flowers --suite lr --dataset flowers --seeds 0 1 2
batch ablation_size_flowers --suite image-size --dataset flowers --seeds 0 1 2
batch ablation_depth_eurosat --suite depth --dataset eurosat --seeds 0 1 2
echo "=== $(date +%FT%T) idle queue finished"
