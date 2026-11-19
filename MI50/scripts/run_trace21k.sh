#!/bin/bash
# run_trace21k.sh [outdir] -- bring up the 8013 test server under rocprofv3 kernel-level tracing,
# so one prefill request is captured dispatch-by-dispatch (name, grid, workgroup, LDS, VGPR).
#
# Profile is NOT serialised by --kernel-trace in principle, but rocprofiler-sdk still drains the
# dispatch queue, so each kernel's recorded duration is inflated ~10x while its SHARE stays usable.
# Never reuses an existing output dir: a previous trace is data, not scratch.
set -u
PORT=8013
OUT=${1:-/home/yanyq/trace/$(date +%Y%m%d-%H%M%S)}
WD=/home/yanyq/port/mi50-0214
PY=/home/yanyq/venv-strata/bin/python
LOG=/home/yanyq/bench/mi50-0214/logs/serve-trace21k.log

if [ -e "$OUT" ]; then
  echo "refusing to reuse existing trace dir: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
cd "$WD" || exit 1
: > "$LOG"

setsid env \
  ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
  STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
  STRATA_HIP_PROMPT_F16=1 \
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \
  STRATA_I8_COUNT=1 STRATA_PREFILL_TIMING=1 \
  STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1 \
  taskset -c 0-15 \
  /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -o "$OUT/run" -- \
  "$PY" -m serve.server --engine strata --config /home/yanyq/bench/cfg-mi50-i8tune.json \
  --port "$PORT" --host 127.0.0.1 \
  --tokenizer /data480/models/strata-pack-ud31-q4kxl/tokenizer > "$LOG" 2>&1 &

echo "launched: port $PORT  log $LOG  out $OUT"
