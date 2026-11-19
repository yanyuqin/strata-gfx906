#!/bin/bash
# run_mi50_0216.sh [port] -- the MI50 launcher on the v0.2.16 sync tree (~/port/mi50-0216).
# Same golden environment as run_mi50_prod.sh (measured 2026-11-17, 21K prefill 941.8 tok/s); only the
# tree layout changed: the engine sources live in engine/, the server in terminal/serve/.
#   ROCR_VISIBLE_DEVICES=0,2   keep the RX 7900 XTX (device 1) out of the HIP set
#   STRATA_HIP_PROMPT_F16=1    FP16 prompt GEMMs (+15~17% prefill; gfx906 is still not in 0.2.16's default list)
#   STRATA_I8_TM=2             tile 2 (the 2-blocks/CU occupancy target; tile 4 spills 27 VGPRs: 783 tok/s)
#   STRATA_I8_N8=0             16-entry slices
#   STRATA_I8_PF=1             one stage of prefetch (PF=2 spills at tile 2: 863.9 against 946.3)
#   taskset -c 0-15            the 16 physical cores
set -uo pipefail
PORT=${1:-8013}
WD=/home/yanyq/port/mi50-0216
PY=/home/yanyq/venv-strata/bin/python
CFG=${CFG:-/home/yanyq/bench/cfg-mi50-0216.json}
cd "$WD" || exit 1
taskset -c 0-15 setsid env \
  ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
  PYTHONPATH="$WD/terminal" \
  STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
  STRATA_HIP_PROMPT_F16=1 \
  STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1 \
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \
  "$PY" -m serve.server --engine strata --config "$CFG" \
  --port "$PORT" --host 127.0.0.1 \
  --tokenizer /data480/models/strata-pack-ud31-q4kxl/tokenizer
