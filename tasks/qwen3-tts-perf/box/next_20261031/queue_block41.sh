#!/usr/bin/env bash
# Block 41 of 10-31: the compile PR rebuilt on main (COMP: perf/qwen3-tts-reference-encoder-one-compile) against main
# (MAIN, which now carries #3368 and #3380). Design: encoder_compile_20261031/DESIGN.md.
# (1) SeedTTS EN references staged. (2) card 0: the one-program GPU test under TORCH_LOGS=recompiles_verbose, which names
#     the guard behind the small config's second graph, then the same test with every codebook table built before the
#     runner (tables_first_plugin.py): if the second graph is the lazily built table, the test passes there; card 1 at
#     the same time: the qwen3_tts suite on COMP. (3) card 0 alone: block 39's compile forms, each cold (fresh
#     TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR, all 1,088 references) then warm (50): MAIN as served (no compile),
#     per-bucket static, regional, the gate's CosyVoice3 form, and COMP as served; the compile workers' count and summed
#     RSS sampled every second through each cold run. (4) boots, MAIN card 0 and COMP card 1 at once, each tree on its own
#     fresh inductor and Triton caches: default 1.7B Base cold then warm, 1.7B CustomVoice, the cascade profiles (0.6B
#     static 0.08, 1.7B static 0.12): health time, KV pool, encoder graph lines. (5) ab_runs.py, the standard SeedTTS A/B.
# Ports 17000 to 19900 for boots and 11000 for the A/B, refused when one already answers.
# usage: D=... T=<kit tree> MAIN=<tree> COMP=<tree> bash queue_block41.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
COMP=${COMP:?the compile tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block41
L=$W/queue.log
GPU_TEST=tests/unit_test/qwen3_tts/test_reference_encoder_cuda_graph.py::test_reference_encoder_graph_replays_match_the_compiled_encode_and_miss_above_the_largest_bucket
mkdir -p $W/logs
for tree in $MAIN $COMP; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni tests || echo DIRTY)" >> $L
done
echo "queue block41 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

(cd $MAIN && PYTHONPATH=$MAIN python3 $K/bisect_20261027/stage_seedtts_wavs.py $W/wavs > $W/logs/stage.log 2>&1)
echo "wavs: $(tail -n 1 $W/logs/stage.log) $(date +%T)" >> $L

(cd $COMP && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$COMP TORCH_LOGS=recompiles_verbose python3 -m pytest -q -s \
  -p no:cacheprovider $GPU_TEST > $W/logs/gpu_test_as_is.log 2>&1
 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$COMP:$K/next_20261031 TORCH_LOGS=recompiles_verbose python3 -m pytest -q -s \
  -p no:cacheprovider -p tables_first_plugin $GPU_TEST > $W/logs/gpu_test_tables_first.log 2>&1) &
a=$!
(cd $COMP && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$COMP timeout 3600 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/qwen3_tts > $W/logs/suite.log 2>&1) &
b=$!
wait $a $b
for run in as_is tables_first; do
  echo "gpu test $run: $(grep -E 'passed|failed' $W/logs/gpu_test_$run.log | tail -n 1); $(grep -m 1 -E '^E +assert' $W/logs/gpu_test_$run.log)" >> $L
  grep -E -A4 'Recompiling function|guard failure' $W/logs/gpu_test_$run.log | head -n 20 | sed "s/^/  $run | /" >> $L
done
echo "suite: $(tail -n 1 $W/logs/suite.log) $(grep -E '^(FAILED|ERROR) ' $W/logs/suite.log | head -n 5 | tr '\n' ' ') $(date +%T)" >> $L

sample_workers() {
  while true; do
    echo "$(date +%T) $(ps -eo args | grep -c '[c]ompile_worker') $(ps -eo rss,args | awk '/[c]ompile_worker/ {sum += $1} END {printf "%.0f", sum / 1024}')" >> $1
    sleep 1
  done
}

gate() {
  local name=$1 tree=$2 variant=$3 phase=$4 clips=$5 sampler=
  if [ $phase = cold ]; then
    sample_workers $W/logs/workers_$name.txt &
    sampler=$!
  else
    :
  fi
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_$name/inductor TRITON_CACHE_DIR=$W/cache_$name/triton \
    TORCH_LOGS=recompiles python3 $K/next_20261029/encoder_speedups_gate.py $W/wavs --model $M17 --card 0 \
    --variants $variant --clips $clips > $W/${name}_$phase.txt 2>&1)
  if [ -n "$sampler" ]; then
    kill $sampler
  else
    :
  fi
  echo "$name $phase: $(grep -E "^== $variant" $W/${name}_$phase.txt | cut -c1-170) | $(grep -E 'official / graphs' $W/${name}_$phase.txt | tr -s ' ') | recompiles $(grep -c '\[__recompiles\] Recompiling' $W/${name}_$phase.txt) $(date +%T)" >> $L
}

for spec in "main_served $MAIN served" "static $MAIN compile" "regional $MAIN compile_regional" "cosy $MAIN compile_cosy" "pr $COMP served"; do
  set -- $spec
  rm -rf $W/cache_$1
  gate $1 $2 $3 cold 1088
  if [ -s $W/logs/workers_$1.txt ]; then
    echo "  $1 compile workers: at most $(awk '{print $2}' $W/logs/workers_$1.txt | sort -n | tail -n 1) processes, $(awk '{print $3}' $W/logs/workers_$1.txt | sort -n | tail -n 1) MiB summed RSS" >> $L
  else
    :
  fi
  gate $1 $2 $3 warm 50
done

boot() {
  local label=$1 tree=$2 port=$3 cache=$4
  shift 4
  if (echo > /dev/tcp/127.0.0.1/$port) 2> /dev/null; then
    echo "$label port $port already answers, not started" >> $L
    return 1
  else
    :
  fi
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" \
    > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 1
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s, $(grep -o 'KV pool holds [0-9]* tokens' $W/logs/serve_$label.log | head -n 1), encoder graph lines $(grep -c 'reference encoder graphs captured' $W/logs/serve_$label.log) $(date +%T)" >> $L
}

stop() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

pair() {
  local name=$1 port=$2
  shift 2
  boot ${name}_main $MAIN $port $W/boot_cache_main --tts_engine.gpu 0 --vocoder.gpu 0 $* &
  local a=$!
  boot ${name}_comp $COMP $((port + 200)) $W/boot_cache_comp --tts_engine.gpu 1 --vocoder.gpu 1 $* &
  local b=$!
  wait $a $b
  stop ${name}_main
  stop ${name}_comp
}

rm -rf $W/boot_cache_main $W/boot_cache_comp
pair cold 17000 --model-path $M17
pair warm 17600 --model-path $M17
pair customvoice 18200 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
pair cascade06 18800 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08
pair cascade17 19400 --model-path $M17 --config examples/configs/qwen3_tts_1_7b.yaml --mem-fraction-static 0.12

(cd $K/next_20261031 && python3 ab_runs.py --a main=$MAIN --b comp=$COMP --root $W/ab --port 11000 > $W/ab.txt 2>&1)
echo "ab: $(grep -E '^(n16|s16)_o[12]|paired' $W/ab.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block41 done $(date +%T)" >> $L
