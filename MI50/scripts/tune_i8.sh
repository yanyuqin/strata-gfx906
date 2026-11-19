#!/bin/bash
# tune_i8.sh -- restart the 8013 test server per (STRATA_I8_TM, STRATA_I8_N8, STRATA_I8_PF) cell, run one 21K prompt,
# and print the chunk's GPU timeline with the gate/up phase.  Cells are TM:N8:PF.
set -u
PORT=8013
LOG=/home/yanyq/bench/engine-i8tune.log

for cell in "$@"; do
  TM=${cell%%:*}
  rest=${cell#*:}
  N8=${rest%%:*}
  PF=${rest##*:}
  I8EXTRA="STRATA_I8_TM=$TM STRATA_I8_N8=$N8 STRATA_I8_PF=$PF" bash /home/yanyq/bench/run_i8tune.sh "$PORT" >/dev/null
  for i in $(seq 1 100); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    sleep 3
  done
  line=$(python3 /home/yanyq/bench/bench40k.py --port "$PORT" --tokens 21000 --predict 32 --reps 1 --label "tm${TM}n8${N8}pf${PF}" 2>&1 | tail -2 | head -1)
  echo "=== cell TM=$TM N8=$N8 PF=$PF : $line"
  grep -a "strata prefill timing: 1[0-9]* tokens" "$LOG" | tail -1 | sed -e 's/.*GPU timeline/GPU timeline/' -e 's/, wall.*//'
  pkill -f "[s]erve.server --engine strata --config /home/yanyq/bench/cfg-mi50-i8tune.json --port $PORT" 2>/dev/null
  pkill -f "[s]trata --serve" 2>/dev/null
  for i in $(seq 1 20); do
    if ! ss -ltn 2>/dev/null | grep -q ":$PORT "; then break; fi
    sleep 2
  done
done
echo DONE
