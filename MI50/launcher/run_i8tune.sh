#!/bin/bash
# run_i8tune.sh [port] -- MI50 test launcher for the i8 expert-kernel tile tuning round.
# Same environment as run_mi50_prod.sh (including the golden STRATA_I8_TM=2 N8=0 PF=1), plus the i8 diagnostics;
# I8EXTRA comes last and overrides any of them for an arm.
PORT=${1:-8013}
WD=/home/yanyq/port/mi50-0214
PY=/home/yanyq/venv-strata/bin/python
LOG=/home/yanyq/bench/log-i8tune-${PORT}.log
cd "$WD" || exit 1
: > "$LOG"
setsid env \
  ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
  STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
  STRATA_HIP_PROMPT_F16=1 \
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \
  STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1 \
  STRATA_I8_COUNT=1 STRATA_PREFILL_TIMING=1 ${I8EXTRA:-} \
  taskset -c 0-15 \
  "$PY" -m serve.server --engine strata --config /home/yanyq/bench/cfg-mi50-i8tune.json \
  --port "$PORT" --host 127.0.0.1 \
  --tokenizer /data480/models/strata-pack-ud31-q4kxl/tokenizer > "$LOG" 2>&1 &
echo "started port $PORT log $LOG"
