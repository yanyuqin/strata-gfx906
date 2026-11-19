#!/bin/bash
# verify_final.sh -- start the recommended MI50 config once on port 8013, measure, stop.
# Proves the shipped cfg-mi50-final.json boots and serves; 8000 stays untouched.
set -uo pipefail
PORT=8013
CFG=/home/yanyq/bench/cfg-mi50-final.json
WD=/home/yanyq/port/mi50-0214
PY=/home/yanyq/venv-strata/bin/python
SLOG=/home/yanyq/bench/mi50-0214/results/verify-serve.log
: > "$SLOG"

for x in $(pgrep -f "serve.server --engine strata.*--po[r]t $PORT" || true); do kill -TERM "$x" 2>/dev/null; done
sleep 3

t0=$(date +%s)
cd "$WD" || exit 1
taskset -c 0-15 setsid env \
  ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
  STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
  STRATA_HIP_PROMPT_F16=1 \
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \
  "$PY" -m serve.server --engine strata --config "$CFG" \
  --port "$PORT" --host 127.0.0.1 --tokenizer /data480/models/strata-pack-ud31-q4kxl/tokenizer \
  < /dev/null > "$SLOG" 2>&1 &

ok=0
for i in $(seq 1 180); do
  grep -q "^ready: " "$SLOG" 2>/dev/null && { ok=1; break; }
  pgrep -f "serve.server --engine strata.*--po[r]t $PORT" >/dev/null || { echo "server exited"; tail -20 "$SLOG"; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "not ready"; tail -20 "$SLOG"; exit 1; }
echo "ready=$(($(date +%s)-t0))s vram card1=$(($(cat /sys/class/drm/card1/device/mem_info_vram_used)/1048576)) card3=$(($(cat /sys/class/drm/card3/device/mem_info_vram_used)/1048576)) MiB"
grep -E "expert cache [0-9]+ slots|resident RAM mode|PLE table locked" "$SLOG" | head -3

python3 /home/yanyq/bench/mi50-0214/mi50_bench.py --port $PORT --case 5k --reps 2 --gen 64 --tag final-check-5k
python3 /home/yanyq/bench/mi50-0214/mi50_bench.py --port $PORT --case 21k --reps 1 --gen 64 --tag final-check-21k

for x in $(pgrep -f "serve.server --engine strata.*--po[r]t $PORT" || true); do kill -TERM "$x" 2>/dev/null; done
sleep 5
for x in $(pgrep -f "serve.server --engine strata.*--po[r]t $PORT" || true); do kill -KILL "$x" 2>/dev/null; done
for i in $(seq 1 60); do u=$(($(cat /sys/class/drm/card1/device/mem_info_vram_used)/1048576)); [ "$u" -lt 200 ] && break; sleep 2; done
echo "stopped, card1 vram=${u} MiB"
