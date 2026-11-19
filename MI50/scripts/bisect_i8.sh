#!/bin/bash
# bisect_i8.sh -- the i8 expert kernel's cost breakdown by the kernel's own bisect points
# (STRATA_I8_BISECT=n returns at point n, uniform over the block), one restart per cell.
set -u
PORT=8013
LOG=/home/yanyq/bench/engine-i8tune.log

for cell in "$@"; do
  EXTRA=${cell%%|*}
  LABEL=${cell##*|}
  I8EXTRA="$EXTRA" bash /home/yanyq/bench/run_i8tune.sh "$PORT" >/dev/null
  ok=0
  for i in $(seq 1 100); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ok=1; break; fi
    sleep 3
  done
  if [ "$ok" != 1 ]; then echo "=== $LABEL: server did not come up"; pkill -f "[s]erve.server.*8013"; pkill -f "[s]trata --serve"; sleep 6; continue; fi
  out=$(python3 /home/yanyq/bench/bench40k.py --port "$PORT" --tokens 21000 --predict 32 --reps 1 --label "$LABEL" 2>&1 | tail -2 | head -1)
  echo "=== $LABEL ($EXTRA): $out"
  grep -a "GPU timeline" "$LOG" | grep "167[0-9][0-9] tokens" | tail -1 | grep -o "GPU timeline [0-9]* ms\|gemm gate/up [0-9]* ([0-9.]*%)\|gdn [0-9]* ([0-9.]*%)\|qsa attn [0-9]* ([0-9.]*%)"
  pkill -f "[s]erve.server.*8013" 2>/dev/null
  pkill -f "[s]trata --serve" 2>/dev/null
  sleep 6
done
echo DONE
