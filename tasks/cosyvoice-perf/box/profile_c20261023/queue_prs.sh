#!/usr/bin/env bash
# 10-23, cards 0 and 1: each PR on its own pair on the base it lands on, every cell warmed at its concurrency (the
# benchmark's default). Arms: t_mh main (the L4 rework's base), t_l4h the L4 rework (the engine's headroom, an
# explicit size capped, no L2 and no L3), t_l6h L6 on it, t_l5h L5 on that, t_l3e L3 with the event fix on main.
# Units of the stack's top and of L3; L4's worst step without L3 (W16 then stream c64, 16 and 32 rows); the pairs
# p_h4 (main, L4), p_h6 (L4, L6), p_h5 (L6, L5), p_h3 (main, L3) at c1f, c16, c64, b1f, b16, both orders, with each
# second's host load; byte identity of the seeded cells; WER and SIM of every arm's c16, then of p_h5's and p_h3's c1f.
# Decision rules, written before the runs: the worst step passes when every W and c64 request completes with no
# allocator failure and no traceback, else L4 lands on L3 and p_h4 is void; a row moves when both orders agree in
# sign and their mean leaves the cell's A/A band; a PR's cost stands only beside a gain of 8 % or more in its own
# pair.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_prs.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261023/queue_prs.sh'
D=/data/c20261022b
P=$D/prs
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
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
l4h=$(fetch perf/cosyvoice3-prefix-pool-headroom)
l6h=$(fetch perf/cosyvoice3-prefix-graph-ladder-v2)
l5h=$(fetch perf/cosyvoice3-prefill-graphs-v3)
l3e=$(fetch perf/cosyvoice3-speech-tokenizer-cuda-graphs)
tree t_mh $(git -C /workspace/sglang-omni rev-parse $l4h~2)
tree t_l4h $l4h
tree t_l6h $l6h
tree t_l5h $l5h
tree t_l3e $l3e
log "queue prs start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
units() {
  local name=$1 card=$2
  rm -rf $P/inductor_units_$name
  (cd /workspace/$name && CUDA_VISIBLE_DEVICES=$card TORCHINDUCTOR_CACHE_DIR=$P/inductor_units_$name \
    PYTHONPATH=/workspace/$name:$COSYVOICE taskset -c $(cpus $card) \
    timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
    > $P/units_$name.log 2>&1)
  log "units $name rc $? $(grep -E 'passed|failed' $P/units_$name.log | tail -n 1)"
}
units t_l5h 0 &
units t_l3e 1 &
wait
grep -q 'units t_l5h rc 0' $P/queue.txt && grep -q 'units t_l3e rc 0' $P/queue.txt || { log "units failed, stop"; exit 1; }
for rows in 16 32; do
  [ -s $D/worst$rows/meta.lst ] || (cd /workspace/t_mh && python3 $IN/make_worst_meta_c20261011.py --out $D/worst$rows \
    --requests $rows > $P/worst_meta_$rows.log 2>&1)
done
rm -rf $D/inductor_t_l4h
python3 $IN/worst_step_c20261021.py /workspace/t_l4h 0 $(cpus 0) 46000 $P/worst_l4h_16 16 $D/worst16/meta.lst > $P/worst_l4h_16.txt 2>&1 &
python3 $IN/worst_step_c20261021.py /workspace/t_l4h 1 $(cpus 1) 46400 $P/worst_l4h_32 32 $D/worst32/meta.lst > $P/worst_l4h_32.txt 2>&1 &
wait
for rows in 16 32; do
  log "worst step l4h $rows rows: $(grep -E 'W[0-9]+:|c64:' $P/worst_l4h_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-360)"
  grep -E 'alloc retries|outside torch|peak|reserve|Flow prefix cache' $P/worst_l4h_$rows.txt | sed 's/^/    /' >> $P/queue.txt
done
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  python3 $IN/host_load_c20261022.py $P/${name}_load.jsonl &
  local sampler=$!
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $P/$name > $P/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $P/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  kill $sampler
  python3 $IN/host_load_c20261022.py --summary $P/${name}_load.jsonl $P/$name $cells > $P/${name}_load.txt 2>&1
}
pair t_mh t_l4h p_h4 c1f,c16,c64,b1f,b16
pair t_l4h t_l6h p_h6 c1f,c16,c64,b1f,b16
pair t_l6h t_l5h p_h5 c1f,c16,c64,b1f,b16
pair t_mh t_l3e p_h3 c1f,c16,c64,b1f,b16
for name in p_h4 p_h6 p_h5 p_h3; do
  python3 $HERE/identity.py $P/$name > $P/identity_$name.txt 2>&1
  log "identity $name $(grep -E 'c1f|b1f' $P/identity_$name.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
done
score() {
  local dir=$1 card=$2 port=$3
  if [ ! -s $dir/wer_results.json ]; then
    (cd /workspace/t_mh && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
      > $dir/score_wer.log 2>&1)
    log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  fi
  if [ ! -s $dir/similarity_results.json ]; then
    (cd /workspace/t_mh && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
    log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
  fi
}
round=0
for target in p_h4:c16 p_h6:c16 p_h5:c16 p_h3:c16 p_h5:c1f p_h3:c1f; do
  name=${target%%:*}
  cell=${target##*:}
  for order in a b; do
    round=$((round + 1))
    score $P/$name/$order/gpu0/$cell 0 $((26000 + 600 * round)) &
    score $P/$name/$order/gpu1/$cell 1 $((26300 + 600 * round)) &
    wait
  done
done
log "queue prs done"
