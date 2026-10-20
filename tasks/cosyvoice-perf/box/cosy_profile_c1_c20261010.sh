#!/usr/bin/env bash
# c1 only variant of cosy_profile_run.sh for step 1's AR slowdown at c1: the server under nsys
# with the combined probe and the OS runtime trace, a warmup, one traced seeded stream c1
# collection, the server stopped by its process group, then cosy_ledger and ar_loop_ledger.
# usage (in the container): bash cosy_profile_c1_c20261010.sh <tree> <card> <cpus> <port> <out>
TREE=$1 CARD=$2 CPUS=$3 PORT=$4 R=$5
IN=/data/c20261010/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
SESSION=cosyprof_$CARD
mkdir -p $R
cd /workspace/$TREE
echo "start $(date +%T) tree $TREE head $(git rev-parse --short HEAD) card $CARD cpus $CPUS port $PORT" > $R/progress.txt
cp /workspace/$TREE/BRANCH_MARKER.txt $R/ 2>/dev/null
PYTHONPATH=/workspace/$TREE python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/import_path.txt

setsid env CUDA_VISIBLE_DEVICES=$CARD SGLANG_OMNI_STRICT_PORT=1 OMNI_LOOP_PROBE=1 OMNI_PIPE_NVTX=1 \
  TORCHINDUCTOR_CACHE_DIR=/data/c20261010/inductor_prof_$TREE \
  PYTHONPATH=/workspace/$TREE:$IN/cosy_profile_probe \
  taskset -c $CPUS nsys launch --session-new=$SESSION --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  python3 -m sglang_omni.cli serve --model-path $MODEL --host 127.0.0.1 --port $PORT \
  > $R/serve.log 2>&1 < /dev/null &
SERVE_PID=$!
echo "serve pgid $SERVE_PID" >> $R/progress.txt

for i in $(seq 1 360); do
  if curl -sf http://127.0.0.1:$PORT/health > /dev/null; then
    echo "healthy $(date +%T)" >> $R/progress.txt
    break
  fi
  if ! kill -0 $SERVE_PID 2> /dev/null; then
    echo "server died $(date +%T)" >> $R/progress.txt
    exit 1
  fi
  sleep 5
done

bench() {
  local name=$1 concurrency=$2 samples=$3 seed=${4:-}
  (cd /workspace/sglang-omni && taskset -c $CPUS env CUDA_VISIBLE_DEVICES= python3 -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --port $PORT --use-existing-server --generate-only --stream --lang en --warmup 1 \
    --max-concurrency $concurrency --max-samples $samples ${seed:+--seed $seed} \
    --output-dir $R/bench_$name) > $R/bench_$name.log 2>&1
  echo "bench $name rc $? $(date +%T)" >> $R/progress.txt
}

traced() {
  local name=$1
  shift
  nsys start --session=$SESSION --sample=none --cpuctxsw=none --output=$R/$name --force-overwrite=true
  bench "$name" "$@"
  nsys stop --session=$SESSION
}

bench warm 16 64
traced c1 1 24 1234
for i in $(seq 1 90); do
  [ -s $R/c1.nsys-rep ] && break
  sleep 2
done
echo "report $(ls -la $R/c1.nsys-rep 2>&1) $(date +%T)" >> $R/progress.txt
kill -TERM -- -$SERVE_PID
sleep 20
kill -KILL -- -$SERVE_PID 2> /dev/null
echo "server stopped $(date +%T)" >> $R/progress.txt

for cell in c1:1; do
  name=${cell%%:*}
  concurrency=${cell##*:}
  nsys export --type sqlite --force-overwrite=true --output $R/$name.sqlite $R/$name.nsys-rep \
    > $R/export_$name.log 2>&1
  python3 $IN/cosy_ledger.py $R/$name.sqlite --skip-head-s 1 --skip-tail-s 1 > $R/cosy_ledger_$name.txt 2>&1
  python3 /data/box/ar_loop_ledger_v6.py $R/$name.sqlite --skip-head-s 1 --skip-tail-s 1 > $R/ar_ledger_$name.txt 2>&1
  python3 /data/box/stage_ledger_v4.py $R/$name.sqlite --concurrency $concurrency --skip-head-s 1 \
    --skip-tail-s 1 > $R/stage_ledger_$name.txt 2>&1
  echo "ledgers $name $(date +%T)" >> $R/progress.txt
done
echo "done $(date +%T)" >> $R/progress.txt
