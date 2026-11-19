#!/bin/bash
# run_i8verify_nopin.sh [port] -- does the source default hold when nobody sets the i8 knobs?
# Deliberately sets none of STRATA_I8_TM / N8 / PF: the engine picks the tile itself and the source default must be
# PF=1 on gfx906 (PF=2 spills at tile 2 and costs ~24%).  Prints the launch's tile/smem and the three 21K reps.
PORT=${1:-8013}
WD=/home/yanyq/port/mi50-0214
PY=/home/yanyq/venv-strata/bin/python
LOG=/home/yanyq/bench/log-i8verify-${PORT}.log
cd "$WD" || exit 1
: > "$LOG"
setsid env \
  ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
  STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
  STRATA_HIP_PROMPT_F16=1 \
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \
  STRATA_I8_COUNT=1 STRATA_PREFILL_TIMING=1 \
  taskset -c 0-15 \
  "$PY" -m serve.server --engine strata --config /home/yanyq/bench/cfg-mi50-i8tune.json \
  --port "$PORT" --host 127.0.0.1 \
  --tokenizer /data480/models/strata-pack-ud31-q4kxl/tokenizer > "$LOG" 2>&1 &
echo "started port $PORT log $LOG"
