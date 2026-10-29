# One boot of a tree with /dev/shm filled to 200 MiB free once the server is healthy, then one
# Video-MME request (its 346 MB preprocessing payload cannot fit): records what a failed SHM put does.
# usage: failure_probe.sh <tree> <out dir> <card> <pin cpus>
set -u
TREE=$1 OUT=$2 CARD=$3 PIN_CPUS=$4
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
mkdir -p "$OUT"
PORT=$(python3 /data/ab/pick_port.py /data/ab/ports.used)
cd "$TREE" || exit 1
git rev-parse HEAD HEAD:sglang_omni > "$OUT/head.txt"
git diff --quiet HEAD -- sglang_omni || echo DIRTY >> "$OUT/head.txt"
SERVE_ARGS="--config examples/configs/qwen3_omni_colocated_h200.yaml --colocate --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
setsid bash -c "echo \$\$ > $OUT/serve.pgid; exec env CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE numactl --physcpubind=$PIN_CPUS --membind=0 python3 -u -m sglang_omni.cli serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct $SERVE_ARGS --host 127.0.0.1 --port $PORT" > "$OUT/serve.log" 2>&1 &
healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  sleep 5
done
echo "port $PORT healthy=$healthy $(date +%T)" > "$OUT/progress.txt"
if [ $healthy = 1 ]; then
  free_kib=$(df --output=avail -k /dev/shm | tail -1)
  fallocate -l $((free_kib - 204800))K /dev/shm/omni_fill_probe
  df -k /dev/shm | tail -1 >> "$OUT/progress.txt"
  CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$TREE timeout 300 python3 "$S/run_bench.py" gen --arm videomme --port "$PORT" --concurrency 1 --out "$OUT" --max-samples 1 > "$OUT/gen.log" 2>&1
  echo "gen rc $? $(date +%T)" >> "$OUT/progress.txt"
  sleep 20
  echo "health after: $(curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health) $(date +%T)" >> "$OUT/progress.txt"
  ps -o pid,stat,etime,args -g "$(cat "$OUT/serve.pgid")" | cut -c1-160 >> "$OUT/progress.txt"
  rm -f /dev/shm/omni_fill_probe
else
  echo "server not healthy" > "$OUT/FAILED"
fi
kill -TERM -- -"$(cat "$OUT/serve.pgid")" 2>/dev/null
sleep 15
kill -KILL -- -"$(cat "$OUT/serve.pgid")" 2>/dev/null
grep -nE "SIGBUS|Bus error|OSError|No space left|crashed|Traceback|signal|died|exit code" "$OUT/serve.log" | cut -c1-300 | head -60 > "$OUT/errors.txt"
echo "done $(date +%T)" >> "$OUT/progress.txt"
