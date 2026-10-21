#!/usr/bin/env bash
# 10-22, the 0.5.22 container, cards 0 and 1: the next PRs on upstream main. Phase a: trees t_m (upstream main),
# t_oc (the one call commit), t_l1 (#3019), t_l2 (L2 on #3019), each resolved to its branch's latest commit at the
# start; the touched suites; warms; p_r1 t_m against t_oc at b16 (a batch of one has nothing to group); p_r2 t_l1
# against t_l2 at c16 and c64 (L2's 0.5.21 row was c64's gaps), every gap pooled per arm.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022r/queue_r.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_r_c20261022.sh a'
D=/data/c20261022r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
export PAIR_HOME=$D
mkdir -p $D/units
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree_log() { log "tree $1 $(git -C /workspace/$1 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$1 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$1 diff --quiet HEAD -- sglang_omni || echo DIRTY)"; }
tree() {
  local name=$1 remote=$2 branch=$3 sha
  git -C /workspace/sglang-omni fetch -q $remote $branch
  sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  tree_log $name
}
units() {
  local tree=$1 card=$2
  shift 2
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c $(cpus $card) \
    python3 -m pytest -q -p no:cacheprovider "$@" > $D/units/units_$tree.log 2>&1)
  log "units $tree rc $? $(grep -E '[0-9]+ (passed|failed)' $D/units/units_$tree.log | tail -n 1)"
}
warm() {
  local tree=$1 card=$2 port=$3
  bash $IN/warm_c20261021.sh $tree $card $(cpus $card) $port $D/warm_$tree
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
}
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_m && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --host 127.0.0.1 --port $port \
    --lang en --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
pooled() {
  local name=$1 cell=$2 first=$3 second=$4
  python3 $IN/pooled_inter_chunk_c20261021.py $D/$name $cell $first $second > $D/pooled_${name}_$cell.txt 2>&1
  log "pooled $name $cell $(tr '\n' ' ' < $D/pooled_${name}_$cell.txt | cut -c1-900)"
}

if [ "$1" = a ]; then
  log "queue a start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
  tree t_m upstream main
  tree t_oc origin perf/cosyvoice3-buffered-packed-one-call
  tree t_l1 origin perf/cosyvoice3-overlapped-vocoder-steps
  tree t_l2 origin perf/cosyvoice3-expandable-segments-v3
  units t_oc 0 tests/unit_test/fun_cosyvoice3/test_vocoder.py &
  units t_l2 1 tests/unit_test/fun_cosyvoice3/test_pipeline.py &
  wait
  warm t_m 0 50000 &
  warm t_oc 1 50400 &
  wait
  warm t_l1 0 50800 &
  warm t_l2 1 51200 &
  wait
  pair t_m t_oc p_r1 b16
  pair t_l1 t_l2 p_r2 c16,c64
  pooled p_r2 c16 t_l1 t_l2
  pooled p_r2 c64 t_l1 t_l2
  log "queue a done"
elif [ "$1" = b ]; then
  # Phase b: t_l3 (L3 with the capture and import fixes) and t_l4 (L4 with the allocator check), each on the one
  # below; their touched suites on a card; warms; once the CPU unit run in /data/c20261022r/ci has ended, p_r3 t_l2
  # against t_l3 (c1f seeded, c16) with c16 scored, then p_r4 t_l3 against t_l4 (c16, c64), every gap pooled.
  log "queue b start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
  tree t_l3 origin perf/cosyvoice3-speech-tokenizer-graphs
  tree t_l4 origin perf/cosyvoice3-prefix-pool-sized-v2
  units t_l3 0 tests/unit_test/fun_cosyvoice3/test_speech_tokenizer.py tests/unit_test/fun_cosyvoice3/test_utils.py &
  units t_l4 1 tests/unit_test/fun_cosyvoice3/test_vocoder.py tests/unit_test/fun_cosyvoice3/test_pipeline.py \
    tests/unit_test/fun_cosyvoice3/test_flow_batch.py &
  wait
  warm t_l3 0 51600 &
  warm t_l4 1 52000 &
  wait
  while pgrep -f "pytest tests/ -q" > /dev/null; do sleep 30; done
  log "cpu unit run ended"
  pair t_l2 t_l3 p_r3 c1f,c16
  pooled p_r3 c16 t_l2 t_l3
  score $D/p_r3/a/gpu0/c16 0 26000 &
  score $D/p_r3/a/gpu1/c16 1 27000 &
  wait
  score $D/p_r3/b/gpu0/c16 0 28000 &
  score $D/p_r3/b/gpu1/c16 1 29000 &
  wait
  pair t_l3 t_l4 p_r4 c16,c64
  pooled p_r4 c16 t_l3 t_l4
  pooled p_r4 c64 t_l3 t_l4
  log "queue b done"
elif [ "$1" = score_r1 ]; then
  # Phase score_r1: p_r1's b16 audio scored, WER and SIM per arm.
  log "queue score_r1 start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
  score $D/p_r1/a/gpu0/b16 0 26000 &
  score $D/p_r1/a/gpu1/b16 1 27000 &
  wait
  score $D/p_r1/b/gpu0/b16 0 28000 &
  score $D/p_r1/b/gpu1/b16 1 29000 &
  wait
  log "queue score_r1 done"
else
  log "unknown phase $1"
fi
