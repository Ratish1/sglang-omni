#!/usr/bin/env bash
# 10-24, host-85-234-79-221, cards 0 and 1: L6 and L5 each on its own pair on the head it opens from, every arm on main
# cd191df23. L6 (perf/cosyvoice3-prefix-graph-ladder-v2, main merged) against L4 (#3129's head, t_l4p), L5
# (perf/cosyvoice3-prefill-graphs-v3, main merged) against L6. Steps: trees; units of each head; a warm of each new
# tree; pairs p_l6c and p_l5c (c1f, c16, c64, b1f, b16, both orders, unprofiled, with each second's host load) and
# their readers (pooled gaps, gap positions, byte identity, tails by audio); WER and SIM of each pair's c16 and c1f;
# then one nsys probe order per pair, both arms at once (c16 and c64 traced), for each PR's mechanism.
# Decision rules, written before the runs: a row moves when both orders agree in sign and their mean leaves the c16
# A/A band of p_aac (3 to 6 % on tail rows); a PR's small cost stands only beside a gain of 8 % or more in its own pair.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l65/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l65.sh'
D=/data/c20261024
P=$D/l65
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261024
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
CELLS=c1f,c16,c64,b1f,b16
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree() {
  local name=$1 sha=$2
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
}
fetch() {
  git -C /workspace/sglang-omni fetch -q origin $1
  git -C /workspace/sglang-omni rev-parse FETCH_HEAD
}
tree t_l4p $(fetch perf/cosyvoice3-prefix-pool-headroom)
tree t_l6p $(fetch perf/cosyvoice3-prefix-graph-ladder-v2)
tree t_l5p $(fetch perf/cosyvoice3-prefill-graphs-v3)
log "queue l65 start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"

units() {
  local name=$1 card=$2
  (cd /workspace/$name && CUDA_VISIBLE_DEVICES=$card TORCHINDUCTOR_CACHE_DIR=$P/inductor_units_$name \
    PYTHONPATH=/workspace/$name:$COSYVOICE taskset -c $(cpus $card) \
    timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
    > $P/units_$name.log 2>&1)
  log "units $name rc $? $(grep -E 'passed|failed' $P/units_$name.log | tail -n 1)"
}
units t_l6p 0 &
units t_l5p 1 &
wait
grep -q 'units t_l6p rc 0' $P/queue.txt && grep -q 'units t_l5p rc 0' $P/queue.txt || { log "units failed, stop"; exit 1; }
bash $IN/warm_c20261021.sh t_l6p 0 $(cpus 0) 29400 $P/warm_t_l6p &
bash $IN/warm_c20261021.sh t_l5p 1 $(cpus 1) 29700 $P/warm_t_l5p &
wait
for name in t_l6p t_l5p; do
  log "warm $name $(grep 'bench rc' $P/warm_$name/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_$name/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_$name/serve.log)"
  grep -E 'Flow prefix cache|prefill.*(graph|backend)|cuda_graph_backend_prefill|Capture.*prefill' $P/warm_$name/serve.log | sed 's/^.*\] //' | cut -c1-200 | sed 's/^/    /' >> $P/queue.txt
done

pair() {
  local first=$1 second=$2 name=$3
  python3 $IN/host_load_c20261022.py $P/${name}_load.jsonl &
  local sampler=$!
  (cd /workspace/sglang-omni && PAIR_CELLS=$CELLS python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $P/$name > $P/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $P/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  kill $sampler
  python3 $IN/host_load_c20261022.py --summary $P/${name}_load.jsonl $P/$name $CELLS > $P/${name}_load.txt 2>&1
  for cell in c16 c64; do
    python3 $IN/pooled_inter_chunk_c20261021.py $P/$name $cell $first $second > $P/${name}_pooled_$cell.txt 2>&1
    python3 $IN/gap_positions_c20261022.py $P/$name $cell > $P/${name}_positions_$cell.txt 2>&1
  done
  python3 $IN/profile_c20261022/identity.py $P/$name > $P/identity_$name.txt 2>&1
  log "identity $name $(grep -E 'c1f|b1f' $P/identity_$name.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
  python3 $HERE/latency_tail_by_audio.py $P/$name b16 c16 c64 > $P/${name}_tails.txt 2>&1
}
pair t_l4p t_l6p p_l6c
pair t_l6p t_l5p p_l5c

score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_mn && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_mn && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
round=0
for target in p_l6c:c16 p_l6c:c1f p_l5c:c16 p_l5c:c1f; do
  name=${target%%:*}
  cell=${target##*:}
  for order in a b; do
    round=$((round + 1))
    score $P/$name/$order/gpu0/$cell 0 $((33000 + 600 * round)) &
    score $P/$name/$order/gpu1/$cell 1 $((33300 + 600 * round)) &
    wait
  done
done

nsys_pair() {
  local first=$1 second=$2 label=$3
  CELLS="c16:16:320 c64:64:320" PYSPY=0 PEER=$P/nsys_${label}_$second \
    bash $IN/cosy_profile_run_c20261021.sh $first 0 $(cpus 0) 47000 $P/nsys_${label}_$first &
  CELLS="c16:16:320 c64:64:320" PYSPY=0 PEER=$P/nsys_${label}_$first \
    bash $IN/cosy_profile_run_c20261021.sh $second 1 $(cpus 1) 47400 $P/nsys_${label}_$second &
  wait
  log "nsys $label $first $(tail -n 1 $P/nsys_${label}_$first/progress.txt) $second $(tail -n 1 $P/nsys_${label}_$second/progress.txt)"
  for cell in c16 c64; do
    python3 $IN/request_states_c20261019.py base=$P/nsys_${label}_$first/$cell.sqlite arm=$P/nsys_${label}_$second/$cell.sqlite \
      --skip-head-s 1 --skip-tail-s 1 > $P/states_${label}_$cell.txt 2>&1
    python3 $IN/chunk_timeline_c20261018.py base=$P/nsys_${label}_$first/$cell.sqlite arm=$P/nsys_${label}_$second/$cell.sqlite \
      --skip-head-s 1 > $P/chunks_${label}_$cell.txt 2>&1
  done
}
nsys_pair t_l4p t_l6p l6
nsys_pair t_l6p t_l5p l5
log "queue l65 done"
