#!/bin/bash
# run_eval_pfpk.sh -- start the 8013 test server on cfg-mi50-prefill3072.json with the PF build, for eval100 checks.
# The config matches the opt1117 baseline run (kv-resident 32768, prefill 3072), so the jsonl can be compared per id.
set -uo pipefail
PORT=8013
CFG=/home/yanyq/bench/cfg-mi50-prefill3072.json
WD=/home/yanyq/port/mi50-0214
PY=/home/yanyq/venv-strata/bin/python
SLOG=/home/yanyq/bench/mi50-0214/logs/serve-eval-pfpk.log
ELOG=/home/yanyq/bench/mi50-0214/logs/engine-mi50-3072.log

for x in $(pgrep -f "serve.server --engine strata.*--port $PORT" || true); do kill -TERM "$x" 2>/dev/null; done
sleep 3
for x in $(pgrep -f "serve.server --engine strata.*--port $PORT" || true); do kill -KILL "$x" 2>/dev/null; done

: > "$SLOG"
: > "$ELOG"
cd "$WD" || exit 1
taskset -c 0-15 setsid env \
  ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
  STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
  STRATA_HIP_PROMPT_F16=1 \
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \
  STRATA_I8_PF=2 \
  "$PY" -m serve.server --engine strata --config "$CFG" \
  --port "$PORT" --host 127.0.0.1 \
  --tokenizer /data480/models/strata-pack-ud31-q4kxl/tokenizer \
  < /dev/null > "$SLOG" 2>&1 &
echo "launched pid=$! port=$PORT"

for i in $(seq 1 180); do
  if grep -q "^ready: " "$SLOG" 2>/dev/null; then echo "ready after ${i}0s"; break; fi
  if ! pgrep -f "serve.server --engine strata.*--port $PORT" >/dev/null; then echo "server exited"; tail -20 "$SLOG"; exit 1; fi
  sleep 5
done
tail -5 "$SLOG"
grep -E "expert cache [0-9]+ slots|KV .*resident|prefill" "$ELOG" | head -4
