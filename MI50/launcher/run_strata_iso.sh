#!/bin/bash
# run_strata_iso.sh <cfg> <port> <tag> <cpuset> <memmax> [hip_devices] [host] [apikey]
# Strata 单实例隔离启动：cgroup v2 MemoryMax 压 page cache + taskset 绑物理核；
# 只按端口清场，不动别的实例。
# host 默认 0.0.0.0（对外服务）；apikey 非空时启用 /v1/* 鉴权。
set -u
CFG=${1:?cfg}
PORT=${2:?port}
TAG=${3:?tag}
CPUSET=${4:?cpuset}
MEMMAX=${5:?memmax}
HVD=${6:-}
HOST=${7:-0.0.0.0}
APIKEY=${8:-}

case "$PORT" in
  8000) TOK=/data480/models/strata-pack-ud-q4kxl/tokenizer ;;
  8093) TOK=/data480/models/strata-pack-ud-iq3xxs/tokenizer ;;
  *)    TOK=/data480/models/strata-pack-ud-q4kxl/tokenizer ;;
esac

LOG=/home/yanyq/bench/serve-$TAG.log
ENGINE_LOG=$(python3 -c "import json,sys;print(json.load(open('$CFG')).get('log',''))")
UNIT=strata-$TAG

pkill -f "serve.server --engine strata.*--po[r]t $PORT" 2>/dev/null
for i in $(seq 1 30); do
  ss -ltn 2>/dev/null | grep -q ":$PORT " || break
  pkill -9 -f "serve.server --engine strata.*--po[r]t $PORT" 2>/dev/null
  sleep 2
done
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "port $PORT busy"; exit 1; }

systemctl --user stop "$UNIT" 2>/dev/null
systemctl --user reset-failed "$UNIT" 2>/dev/null
rm -f "$LOG" "$ENGINE_LOG"

if [ -n "$HVD" ]; then
  ENVSTR="HIP_VISIBLE_DEVICES=$HVD ROCR_VISIBLE_DEVICES=$HVD STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1"
else
  ENVSTR=""
fi

KEYSTR=""
[ -n "$APIKEY" ] && KEYSTR="--api-key $APIKEY"

echo "host=$HOST apikey=${APIKEY:+set}"
systemd-run --user --unit="$UNIT" --collect \
  -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 \
  -p WorkingDirectory=/home/yanyq/strata \
  -p StandardOutput=append:"$LOG" -p StandardError=append:"$LOG" \
  -- taskset -c "$CPUSET" env $ENVSTR \
     /home/yanyq/venv-strata/bin/python -m serve.server --engine strata \
     --config "$CFG" --port "$PORT" --host "$HOST" --tokenizer "$TOK" $KEYSTR
rc=$?
[ $rc -ne 0 ] && { echo "systemd-run failed rc=$rc"; exit 1; }

for i in $(seq 1 300); do
  grep -q "^ready: " "$LOG" 2>/dev/null && { echo "ready after $((i*5))s"; break; }
  systemctl --user is-active --quiet "$UNIT" || { echo "unit died"; tail -40 "$LOG"; exit 1; }
  sleep 5
done

echo "--- cgroup ---"
systemctl --user show "$UNIT" -p MemoryMax -p MemoryCurrent -p MemoryPeak 2>/dev/null | head -5
PID=$(pgrep -f "serve.server --engine strata.*--po[r]t $PORT" | head -1)
echo -n "affinity: "; taskset -pc "$PID" 2>/dev/null
echo "--- bind ---"
ss -ltn 2>/dev/null | grep ":$PORT "
echo "--- engine ---"
grep -E "^ready:|expert cache|resident RAM mode|window up to|KV" "$LOG" | head -8
