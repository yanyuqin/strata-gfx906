#!/usr/bin/env bash
# Start the 8013 test server with the production i8 cell (TM=2 N8=0 PF=1), run three 21000-token prompts
# (bench40k's default --reps 3), print every rep, and the GPU timelines the engine logged for each.
set -u
PORT=8013
LOG=/home/yanyq/bench/engine-i8tune.log
MARK=$(wc -l < "$LOG")

I8EXTRA="STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1" bash /home/yanyq/bench/run_i8tune.sh "$PORT" >/dev/null
for i in $(seq 1 80); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "ready after $((i*3))s"; break; fi
  sleep 3
done
python3 /home/yanyq/bench/bench40k.py --port "$PORT" --tokens 21000 --predict 32 --reps 3 --label rblas-3072 2>&1 | tail -5
echo "--- GPU timeline (this run) ---"
tail -n +$((MARK + 1)) "$LOG" | grep -a "strata prefill timing: 2[0-9]* tokens" | sed -e 's/.*GPU timeline/GPU timeline/' -e 's/, wall.*//'
echo "--- stop ---"
for p in $(ps -eo pid=,comm=,args= | awk '$2=="python" && /serve.server --engine strata/ {print $1}'); do kill -TERM $p 2>/dev/null; done
sleep 5
for p in $(ps -eo pid=,comm=,args= | awk '$2=="python" && /serve.server --engine strata/ {print $1}'); do kill -KILL $p 2>/dev/null; done
sleep 4
ss -ltn | grep -q ":$PORT " && echo "port still busy" || echo "port free"
