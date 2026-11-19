#!/usr/bin/env bash
# Second pass over the 11 rows the first pass wrote as `default` (a solution that wins by 3-4% is below the
# 1.05 threshold but is still a real win).  --min-gain 1.0 keeps the fastest solution whenever it beats
# rocBLAS's default at all; the merge then prefers the row with the larger measured gain.
set -u
GROUP="${1:?usage: $0 A|B}"
BIN=/home/yanyq/port/mi50-0214/build-906-0214/tune_rocblas
OUT=/home/yanyq/bench/gfx906-rocblas-${GROUP}2.txt
LOG=/home/yanyq/bench/tune_rocblas-${GROUP}2-$(date +%H%M).log

if [ "$GROUP" = "A" ]; then
  GPUS=0
  CASES=(
    "2048,10240,2560,10240" "3072,10240,2560,10240"
    "270,12288,2560,12288" "512,12288,2560,12288" "1024,12288,2560,12288"
    "2048,12288,2560,12288" "3072,12288,2560,12288"
  )
else
  GPUS=2
  CASES=(
    "512,2560,2560,2560" "1024,2560,2560,2560" "2048,2560,2560,2560" "3072,2560,2560,2560"
  )
fi

args=()
for c in "${CASES[@]}"; do args+=(--case "$c"); done

echo "pass2 group=$GROUP gpu=$GPUS cases=${#CASES[@]} start=$(date -Is)" | tee "$LOG"
ROCR_VISIBLE_DEVICES=$GPUS "$BIN" "${args[@]}" --min-gain 1.0 --tuning-out "$OUT" 2>&1 | tee -a "$LOG"
echo "pass2 group=$GROUP exit=$? $(date -Is)" | tee -a "$LOG"
