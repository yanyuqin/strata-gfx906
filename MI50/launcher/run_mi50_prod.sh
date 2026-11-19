#!/bin/bash
# run_mi50_prod.sh [port] -- the MI50 production launcher (v0.2.14.1 + FP16 prompt GEMMs + rocBLAS table).
# Golden environment, measured 2026-11-17, 2x MI50 (ROCR 0,2) + EPYC 7F52, 21K prefill 941.8 tok/s:
#   ROCR_VISIBLE_DEVICES=0,2   keep the RX 7900 XTX (device 1) out of the HIP set
#   STRATA_HIP_PROMPT_F16=1    FP16 prompt GEMMs (+15~17% prefill)
#   STRATA_I8_TM=2             the tile the 2-blocks/CU occupancy target is measured at (tile 4 spills 27 VGPRs: 783 tok/s)
#   STRATA_I8_N8=0             16-entry slices; the 8-entry variant measured slower here
#   STRATA_I8_PF=1             one stage of prefetch: PF=2 spills at tile 2 (863.9 tok/s against 946.3; 690.8 at tile 3)
#   taskset -c 0-15            the 16 physical cores (SMT siblings 16-31 left to the scheduler)
set -uo pipefail
PORT=${1:-8013}
WD=/home/yanyq/port/mi50-0214
PY=/home/yanyq/venv-strata/bin/python
cd "$WD" || exit 1
taskset -c 0-15 setsid env \
  ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
  STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
  STRATA_HIP_PROMPT_F16=1 \
  STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1 \
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \
  "$PY" -m serve.server --engine strata --config /home/yanyq/bench/cfg-mi50-final.json \
  --port "$PORT" --host 127.0.0.1 \
  --tokenizer /data480/models/strata-pack-ud31-q4kxl/tokenizer
