#!/usr/bin/env bash
# A/B of the two rocBLAS tables on the same day and thermal state: run three 21000-token prompts with the
# backup (old, T=2048-only) table, print every rep and the engine's per-phase breakdown of the 3072-token
# chunk, then put the new table back and prove it is byte-identical to what it was before the swap.
set -u
B=/home/yanyq/bench
PORT=8013
LOG=$B/engine-i8tune.log

md5sum "$B/gfx906-rocblas.txt" | awk '{print "new table md5 before swap:", $1}'
cp -a "$B/gfx906-rocblas.txt" "$B/gfx906-rocblas-new-20261117.txt"
cp -a "$B/gfx906-rocblas.txt.bak-20261117" "$B/gfx906-rocblas.txt"
grep -c "^f16o" "$B/gfx906-rocblas.txt" | awk '{print "old table rows:", $1}'

MARK=$(wc -l < "$LOG")
I8EXTRA="STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1" bash "$B/run_i8tune.sh" "$PORT" >/dev/null
for i in $(seq 1 90); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "ready after $((i*3))s"; break; fi
  sleep 3
done
python3 "$B/bench40k.py" --port "$PORT" --tokens 21000 --predict 32 --reps 3 --label oldtable 2>&1 | tail -5
echo "--- full request ---"
tail -n +$((MARK + 1)) "$LOG" | grep -a "strata prefill timing: 167" | sed -e 's/host staging.*//'
echo "--- 3072-token chunk ---"
tail -n +$((MARK + 1)) "$LOG" | grep -a "strata prefill timing: 3072 tokens" | tail -1 | sed -e 's/, wall.*//'

for p in $(ps -eo pid=,comm=,args= | awk '$2=="python" && /serve.server --engine strata/ {print $1}'); do kill -TERM $p 2>/dev/null; done
sleep 5
for p in $(ps -eo pid=,comm=,args= | awk '$2=="python" && /serve.server --engine strata/ {print $1}'); do kill -KILL $p 2>/dev/null; done
sleep 3

cp -a "$B/gfx906-rocblas-new-20261117.txt" "$B/gfx906-rocblas.txt"
md5sum "$B/gfx906-rocblas.txt" | awk '{print "new table md5 after restore:", $1}'
grep -c "^f16o" "$B/gfx906-rocblas.txt" | awk '{print "restored rows:", $1}'
ss -ltn | grep -q ":$PORT " && echo "port still busy" || echo "port free"
