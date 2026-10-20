#!/usr/bin/env bash
# Fun-CosyVoice3 profile of one tree, the 10-10 runner (cosy_profile_run.sh) with three changes: graph level CUDA
# graph tracing (node level inflates gaps, KIT_AUDIT 2), this lease's paths and warm inductor caches, and a count of
# every probe label in each trace (the probe skips a renamed target silently, so coverage is checked per run).
# The server runs under nsys with the probe (cosy_profile_probe: loop.*, mr.*, pre.*, voc.*, flow.*, hift.*) and the
# OS runtime trace, a warmup, one traced collection per cell (stream c1 seeded, c16), then py-spy (GIL holder,
# 50 Hz) at c16 without nsys collecting, the server stopped by its process group, and the ledgers read on the box.
# The pipeline pid for py-spy comes from the server log's stage group line. With PEER set to the other arm's output
# folder, the runner waits after its health check until the other arm is healthy too, so both arms trace together.
# EXTRA_PATH goes on PYTHONPATH between the tree and the probe (a measurement patch with its own sitecustomize).
# 10-16 copy: this lease's paths, and the boot identity (head, source tree hash, dirty check, imported file).
# 10-17 copy: the 10-17 lease's paths (/data/c20261017).
# 10-18 copy: this lease's paths (/data/c20261018), the AR ledger with its by rows table, and an optional fifth cell
# field, an open loop request rate (Poisson arrivals from --arrival-seed 2910, so both arms get one schedule).
# usage (in the container): [PEER=<other out>] [EXTRA_PATH=<dir>] bash cosy_profile_run_c20261018.sh <tree> <card>
#   <cpus> <port> <out>
TREE=$1 CARD=$2 CPUS=$3 PORT=$4 R=$5
IN=/data/c20261018/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
SESSION=cosyprof_$CARD
mkdir -p $R
cd /workspace/$TREE
echo "start $(date +%T) tree $TREE head $(git rev-parse --short HEAD) card $CARD cpus $CPUS port $PORT" > $R/progress.txt
{
  git rev-parse HEAD HEAD:sglang_omni
  git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY
  PYTHONPATH=/workspace/$TREE python3 -c "import sglang_omni; print(sglang_omni.__file__)"
} > $R/identity.txt 2>&1

setsid env CUDA_VISIBLE_DEVICES=$CARD SGLANG_OMNI_STRICT_PORT=1 OMNI_LOOP_PROBE=1 OMNI_PIPE_NVTX=1 \
  TORCHINDUCTOR_CACHE_DIR=/data/c20261018/inductor_$TREE \
  PYTHONPATH=/workspace/$TREE:${EXTRA_PATH:+$EXTRA_PATH:}$IN/cosy_profile_probe \
  taskset -c $CPUS nsys launch --session-new=$SESSION --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=graph \
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
touch $R/ready
if [ -n "$PEER" ]; then
  for i in $(seq 1 120); do
    [ -e $PEER/ready ] && break
    sleep 5
  done
  echo "peer ready $(date +%T)" >> $R/progress.txt
fi

bench() {
  local name=$1 concurrency=$2 samples=$3 seed=${4:-} rate=${5:-}
  (cd /workspace/sglang-omni && taskset -c $CPUS env CUDA_VISIBLE_DEVICES= python3 -m benchmarks.eval.benchmark_tts_seedtts \
    --model $MODEL --port $PORT --use-existing-server --generate-only --stream --lang en --warmup 1 \
    --max-concurrency $concurrency --max-samples $samples ${seed:+--seed $seed} \
    ${rate:+--request-rate $rate --arrival-seed 2910} \
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

# CELLS: name:concurrency:samples[:seed[:rate]], traced in order; PYSPY=0 skips the py-spy pass.
CELLS=${CELLS:-c1:1:24:1234 c16:16:320}
bench warm 16 64
for cell in $CELLS; do
  IFS=: read -r name concurrency samples seed rate <<< "$cell"
  traced $name $concurrency $samples "$seed" "$rate"
done

PIPE_PID=$(sed -n 's/.*StageGroup pipeline: spawned 1 process(es) (pids=\[\([0-9]*\)\]).*/\1/p' $R/serve.log | head -1)
echo "pipeline pid ${PIPE_PID:-none}" >> $R/progress.txt
if [ "${PYSPY:-1}" = 1 ]; then
  bench pyspy 16 320 &
  BENCH_PID=$!
  sleep 6
  if [ -n "$PIPE_PID" ]; then
    timeout 60 py-spy record --pid $PIPE_PID --rate 50 --duration 20 --gil --threads \
      --format raw --output $R/pyspy_c16.txt > $R/pyspy.log 2>&1
    echo "pyspy rc $? $(date +%T)" >> $R/progress.txt
  fi
  wait $BENCH_PID
fi
kill -TERM -- -$SERVE_PID
sleep 20
kill -KILL -- -$SERVE_PID 2> /dev/null
echo "server stopped $(date +%T)" >> $R/progress.txt
grep -E "cosy_nvtx|loop_probe" $R/serve.log | sort | uniq -c > $R/probe_log.txt

for cell in $CELLS; do
  IFS=: read -r name concurrency samples seed rate <<< "$cell"
  nsys export --type sqlite --force-overwrite=true --output $R/$name.sqlite $R/$name.nsys-rep \
    > $R/export_$name.log 2>&1
  python3 $IN/nvtx_label_counts_c20261013.py $R/$name.sqlite > $R/labels_$name.txt 2>&1
  python3 $IN/cosy_ledger.py $R/$name.sqlite --skip-head-s 1 --skip-tail-s 1 > $R/cosy_ledger_$name.txt 2>&1
  python3 $IN/ar_loop_ledger_c20261018.py $R/$name.sqlite --skip-head-s 1 --skip-tail-s 1 > $R/ar_ledger_$name.txt 2>&1
  python3 $IN/stage_ledger.py $R/$name.sqlite --concurrency $concurrency --skip-head-s 1 \
    --skip-tail-s 1 > $R/stage_ledger_$name.txt 2>&1
  echo "ledgers $name $(date +%T)" >> $R/progress.txt
done
if [ -s $R/pyspy_c16.txt ]; then
  python3 $IN/pyspy_thread_profile.py $R/pyspy_c16.txt > $R/pyspy_profile_c16.txt 2>&1
  python3 $IN/pyspy_stage_profile.py $R/pyspy_c16.txt --thread-has run_step \
    --phases run_step,select_step_participants,ingest,hop_batch_with_prefix,hop_batch_prefix,leftover_batch,hift_step,hift_group,prepare_flow_conditioning,pack_flow_inputs,split_generated_mels,run,emit_chunks \
    --expand run_step,hop_batch_with_prefix,hop_batch_prefix,leftover_batch,hift_step,run \
    > $R/pyspy_vocoder_c16.txt 2>&1
fi
echo "done $(date +%T)" >> $R/progress.txt
