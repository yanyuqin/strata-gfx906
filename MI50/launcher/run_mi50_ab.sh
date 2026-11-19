#!/bin/bash
# run_mi50_ab.sh [A|B|AB|arm-NAME] [cases] [reps]
#   A = v0.2.10 baseline binary  (~/strata/build-906-0210, port 8010)
#   B = v0.2.14.1 gfx906 build   (~/port/mi50-0214/build-906-0214, port 8011)
#   arm-NAME = cfg-arm-NAME.json (same new binary, tuned args, port 8012)
# Identical for every arm: two MI50s (ROCR 0,2 -> HIP 0,1), taskset 0-15 (16 physical cores),
# the gfx906 env the production launcher uses, same prompt text/seed/token budget.
# The RX 7900 XTX (8093 production) is never touched: only ports 8010-8012 are cleared.
set -uo pipefail

ARM=${1:-AB}
CASES=${2:-"5k 21k"}
REPS=${3:-3}
OUT=/home/yanyq/bench/mi50-0214/results
mkdir -p "$OUT"
STAMP=$(date +%H%M%S)
PY=/home/yanyq/venv-strata/bin/python
BENCH=/home/yanyq/bench/mi50-0214/mi50_bench.py
TOK=/data480/models/strata-pack-ud31-q4kxl/tokenizer

port_of() { case "$1" in A) echo 8010;; B) echo 8011;; *) echo 8012;; esac; }
cfg_of()  { case "$1" in A) echo /home/yanyq/bench/mi50-0214/cfg-base.json;; B) echo /home/yanyq/bench/mi50-0214/cfg-new.json;; *) echo "/home/yanyq/bench/mi50-0214/cfg-arm-$1.json";; esac; }
wd_of()   { case "$1" in A) echo /home/yanyq/strata;; *) echo /home/yanyq/port/mi50-0214;; esac; }

vram_mib() { cat /sys/class/drm/$1/device/mem_info_vram_used 2>/dev/null | awk '{printf "%d", $1/1048576}'; }

stop_arm() {
  local p=$1 i u
  for x in $(pgrep -f "serve.server --engine strata.*--po[r]t $p" || true); do kill -TERM "$x" 2>/dev/null; done
  sleep 4
  for x in $(pgrep -f "serve.server --engine strata.*--po[r]t $p" || true); do kill -KILL "$x" 2>/dev/null; done
  for i in $(seq 1 60); do
    u=$(vram_mib card1); v=$(vram_mib card3)
    [ "${u:-9999}" -lt 200 ] && [ "${v:-9999}" -lt 200 ] && { echo "  stop: vram card1=$u card3=$v MiB"; return 0; }
    sleep 2
  done
  echo "  WARN vram still card1=$u card3=$v MiB"; return 1
}

run_arm() {
  local arm=$1 port cfg wd slog elog t0 ok=0 i
  port=$(port_of "$arm"); cfg=$(cfg_of "$arm"); wd=$(wd_of "$arm")
  slog=$OUT/serve-$arm-$STAMP.log; elog=$(python3 -c "import json;print(json.load(open('$cfg'))['log'])")
  [ -r "$cfg" ] || { echo "FAIL cfg missing $cfg"; return 1; }
  stop_arm "$port"
  : > "$slog"; : > "$elog" 2>/dev/null || true
  echo "===== arm=$arm port=$port cfg=$(basename "$cfg") @ $(date '+%T')"
  t0=$(date +%s)
  cd "$wd" || return 1
  local extra=""
  [ -r "$OUT/env-$arm.env" ] && extra=$(grep -v '^#' "$OUT/env-$arm.env" | tr '\n' ' ')
  taskset -c 0-15 setsid env \
    ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 \
    STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1 \
    $extra \
    "$PY" -m serve.server --engine strata --config "$cfg" \
    --port "$port" --host 127.0.0.1 --tokenizer "$TOK" \
    < /dev/null > "$slog" 2>&1 &
  for i in $(seq 1 180); do
    grep -q "^ready: " "$slog" 2>/dev/null && { ok=1; break; }
    pgrep -f "serve.server --engine strata.*--po[r]t $port" >/dev/null || { echo "FAIL server exited"; tail -30 "$slog"; return 1; }
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "FAIL not ready in 900s"; tail -30 "$slog"; return 1; }
  local ready=$(( $(date +%s) - t0 ))
  local rss=$(awk '/^VmRSS/{printf "%.1f", $2/1048576}' /proc/$(pgrep -x strata-host | head -1)/status 2>/dev/null)
  echo "  ready=${ready}s vram card1=$(vram_mib card1) card3=$(vram_mib card3) MiB rss=${rss} GiB"
  grep -E "expert cache [0-9]+ slots|resident RAM mode|KV .*resident|layer split across|PLE table locked" "$elog" | head -6
  awk '/MemAvailable/{printf "  mem_avail=%.0f GiB\n",$2/1048576}' /proc/meminfo
  {
    echo "ARM $arm ready_s=$ready vram_card1_mib=$(vram_mib card1) vram_card3_mib=$(vram_mib card3) strata_rss_gib=$rss"
    grep -E "expert cache [0-9]+ slots|PLE table locked|KV .*resident|layer split across|resident RAM mode" "$elog" | head -8
  } | tee -a "$OUT/arm-$arm-$STAMP.log"

  local c
  for c in $CASES; do
    echo "--- bench case=$c reps=$REPS"
    "$PY" "$BENCH" --port "$port" --case "$c" --reps "$REPS" --gen 128 --tag "$arm-$c" \
      --out "$OUT/bench-$STAMP.jsonl" 2>&1 | tee -a "$OUT/arm-$arm-$STAMP.log"
  done
  echo "  post: vram card1=$(vram_mib card1) card3=$(vram_mib card3) MiB rss=$(awk '/^VmRSS/{printf "%.1f", $2/1048576}' /proc/$(pgrep -x strata-host | head -1)/status 2>/dev/null) GiB" | tee -a "$OUT/arm-$arm-$STAMP.log"
  stop_arm "$port"
}

case "$ARM" in
  AB) run_arm A; run_arm B;;
  *)  run_arm "$ARM";;
esac
echo "=== done @ $(date '+%T') results: $OUT"
