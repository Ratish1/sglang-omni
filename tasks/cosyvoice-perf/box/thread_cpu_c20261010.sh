#!/usr/bin/env bash
# CPU time per thread of a Fun-CosyVoice3 server, from /proc, over 10 s idle and 10 s of stream c16:
# names the threads that burn cores (ONNX Runtime pools spin after each run by default), which the
# nsys stage ledger cannot see (native threads with no queue waits or CUDA calls).
# usage (in the container): bash thread_cpu_c20261010.sh <tree> <card> <cpus> <port> <out>
TREE=$1 CARD=$2 CPUS=$3 PORT=$4 R=$5
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
mkdir -p $R
cd /workspace/$TREE
setsid env CUDA_VISIBLE_DEVICES=$CARD SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=/workspace/$TREE \
  TORCHINDUCTOR_CACHE_DIR=/data/c20261010/inductor_$TREE \
  taskset -c $CPUS python3 -m sglang_omni.cli serve --model-path $MODEL --host 127.0.0.1 \
  --port $PORT > $R/serve.log 2>&1 < /dev/null &
SERVE_PID=$!
echo "serve pgid $SERVE_PID" > $R/progress.txt
for i in $(seq 1 360); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  kill -0 $SERVE_PID 2> /dev/null || { echo "server died" >> $R/progress.txt; exit 1; }
  sleep 5
done
echo "healthy $(date +%T)" >> $R/progress.txt

snapshot() {
  for pid in $(ps -o pid= -g $SERVE_PID); do
    for task in /proc/$pid/task/*; do
      [ -r $task/stat ] || continue
      tid=${task##*/}
      comm=$(cat $task/comm 2> /dev/null | tr ' ' '_')
      ticks=$(awk '{print $14 + $15}' $task/stat 2> /dev/null)
      echo "$pid $tid $comm $ticks"
    done
  done
}

bench() {
  (cd /workspace/sglang-omni && taskset -c $CPUS env CUDA_VISIBLE_DEVICES= python3 -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --port $PORT --use-existing-server --generate-only --stream --lang en --warmup 1 \
    --max-concurrency 16 --max-samples $1 --output-dir $R/bench_$2) > $R/bench_$2.log 2>&1
}

bench 64 warm
sleep 5
snapshot > $R/idle_a.txt
sleep 10
snapshot > $R/idle_b.txt
bench 640 c16 &
BENCH_PID=$!
sleep 8
snapshot > $R/load_a.txt
sleep 10
snapshot > $R/load_b.txt
wait $BENCH_PID
kill -TERM -- -$SERVE_PID
sleep 20
kill -KILL -- -$SERVE_PID 2> /dev/null
python3 - $R <<'EOF'
import collections
import os
import sys

R = sys.argv[1]
hz = os.sysconf("SC_CLK_TCK")
for phase in ("idle", "load"):
    a = {}
    for line in open(f"{R}/{phase}_a.txt"):
        pid, tid, comm, ticks = line.split()
        a[tid] = int(ticks)
    rows = []
    by_comm = collections.Counter()
    for line in open(f"{R}/{phase}_b.txt"):
        pid, tid, comm, ticks = line.split()
        if tid in a:
            cpu = (int(ticks) - a[tid]) / hz / 10 * 100
            rows.append((cpu, pid, tid, comm))
            by_comm[(pid, comm)] += cpu
    rows.sort(reverse=True)
    print(f"{phase}: total {sum(r[0] for r in rows):.0f} % of one core over 10 s")
    for cpu, pid, tid, comm in rows[:14]:
        print(f"  {cpu:6.1f} %  pid {pid} tid {tid} {comm}")
    print("  by thread name:", ", ".join(f"{c}@{p} {v:.0f}" for (p, c), v in by_comm.most_common(8)))
EOF
echo "done $(date +%T)" >> $R/progress.txt
