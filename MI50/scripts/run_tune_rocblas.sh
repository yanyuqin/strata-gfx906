#!/usr/bin/env bash
# Re-calibrate gfx906-rocblas.txt over the token buckets the engine actually runs.
#   group A (card 1, visible 0): the heavy projections
#   group B (card 3, visible 2): the light ones
# Both run the same 8 buckets per shape; rows are merged by (N,K,ldy,T) afterwards, so which card measured
# a row does not matter (solution indices depend on arch + rocBLAS build only).
set -u
GROUP="${1:?usage: $0 A|B}"
BIN=/home/yanyq/port/mi50-0214/build-906-0214/tune_rocblas
STAMP=$(date +%Y%m%d-%H%M)

if [ "$GROUP" = "A" ]; then
  GPUS=0
  SHAPES=(
    "10240,2560,10240"
    "12288,2560,12288"
    "320,10240,320"
    "10240,320,10240"
    "48,2560,96"
    "48,2560,48"
    "1,2560,1"
  )
else
  GPUS=2
  SHAPES=(
    "6144,2560,6144"
    "2560,6144,2560"
    "2560,640,2560"
    "2560,2560,2560"
    "640,2560,640"
    "512,2560,512"
    "128,2560,128"
    "4,10240,4"
  )
fi
OUT=/home/yanyq/bench/gfx906-rocblas-${GROUP}.txt
LOG=/home/yanyq/bench/tune_rocblas-${GROUP}-${STAMP}.log
TOKENS=(64 128 256 270 512 1024 2048 3072)

args=()
for s in "${SHAPES[@]}"; do
  for t in "${TOKENS[@]}"; do
    args+=(--case "$t,$s")
  done
done

echo "group=$GROUP gpu=$GPUS shapes=${#SHAPES[@]} cases=$(( ${#args[@]} / 2 )) start=$(date -Is)" | tee "$LOG"
ROCR_VISIBLE_DEVICES=$GPUS "$BIN" "${args[@]}" --min-gain 1.05 --tuning-out "$OUT" 2>&1 | tee -a "$LOG"
echo "group=$GROUP exit=$? $(date -Is) rows=$(grep -c '^f16o' "$OUT" 2>/dev/null)" | tee -a "$LOG"
