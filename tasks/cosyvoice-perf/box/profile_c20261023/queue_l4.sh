#!/usr/bin/env bash
# 10-23, cards 0 and 1: the stack L4, L6, L5 on its final L4 (the reserve summed, the declared share honored, the
# tokenizer arena warmed before sizing), each on its own pair on the base it lands on, every cell warmed at its
# concurrency. Waits for queue_h3.sh. The first L4 run (max(headroom, step)) is kept as p_h4_max and worst_l4h_max_*.
# Units of the stack's top; L4's worst step at 16 and 32 rows; pairs p_h4 (main, L4), p_h6 (L4, L6), p_h5 (L6, L5) at
# c1f, c16, c64, b1f, b16, both orders, with each second's host load; byte identity of the seeded cells; WER and SIM of
# every arm's c16, then of p_h5's c1f.
# Decision rules, written before the runs: the worst step passes when every W and c64 request completes with no
# allocator failure and no traceback, else the queue stops; a row moves when both orders agree in sign and their mean
# leaves the cell's A/A band; a PR's cost stands only beside a gain of 8 % or more in its own pair.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_l4.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261023/queue_l4.sh'
D=/data/c20261022b
P=$D/prs
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q 'queue h3 done' $P/queue.txt; do sleep 30; done
for old in p_h4 p_h4.log p_h4_load.jsonl p_h4_load.txt; do
  [ -e $P/$old ] && mv $P/$old $P/${old/p_h4/p_h4_max}
done
for rows in 16 32; do
  [ -e $P/worst_l4h_$rows ] && mv $P/worst_l4h_$rows $P/worst_l4h_max_$rows
  [ -e $P/worst_l4h_$rows.txt ] && mv $P/worst_l4h_$rows.txt $P/worst_l4h_max_$rows.txt
done
tree() {
  local name=$1 sha=$2
  git -C /workspace/$name checkout -q --detach $sha
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-70) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
}
fetch() {
  git -C /workspace/sglang-omni fetch -q origin $1
  git -C /workspace/sglang-omni rev-parse FETCH_HEAD
}
tree t_l4h $(fetch perf/cosyvoice3-prefix-pool-headroom)
tree t_l6h $(fetch perf/cosyvoice3-prefix-graph-ladder-v2)
tree t_l5h $(fetch perf/cosyvoice3-prefill-graphs-v3)
log "queue l4 start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
rm -rf $P/inductor_units_l4
(cd /workspace/t_l5h && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$P/inductor_units_l4 \
  PYTHONPATH=/workspace/t_l5h:$COSYVOICE taskset -c $(cpus 0) \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
  > $P/units_l4.log 2>&1)
units_rc=$?
log "units t_l5h rc $units_rc $(grep -E 'passed|failed' $P/units_l4.log | tail -n 1)"
[ $units_rc = 0 ] || { log "units failed, stop"; exit 1; }
python3 $IN/worst_step_c20261021.py /workspace/t_l4h 0 $(cpus 0) 46000 $P/worst_l4h_16 16 $D/worst16/meta.lst > $P/worst_l4h_16.txt 2>&1 &
python3 $IN/worst_step_c20261021.py /workspace/t_l4h 1 $(cpus 1) 46400 $P/worst_l4h_32 32 $D/worst32/meta.lst > $P/worst_l4h_32.txt 2>&1 &
wait
for rows in 16 32; do
  log "worst step l4h $rows rows: $(grep -E 'W[0-9]+:|c64:' $P/worst_l4h_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-360)"
  grep -E 'alloc retries|outside torch|peak|reserve|Flow prefix cache' $P/worst_l4h_$rows.txt | sed 's/^/    /' >> $P/queue.txt
  grep -q "'failed_requests': 0" $P/worst_l4h_$rows.txt && ! grep -qE 'ooms [1-9]|Traceback' $P/worst_l4h_$rows.txt \
    || { log "worst step $rows rows failed, stop"; exit 1; }
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
for name in p_h4 p_h6 p_h5; do
  python3 $HERE/identity.py $P/$name > $P/identity_$name.txt 2>&1
  log "identity $name $(grep -E 'c1f|b1f' $P/identity_$name.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
done
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_mh && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_mh && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
round=0
for target in p_h4:c16 p_h6:c16 p_h5:c16 p_h5:c1f; do
  name=${target%%:*}
  cell=${target##*:}
  for order in a b; do
    round=$((round + 1))
    score $P/$name/$order/gpu0/$cell 0 $((27000 + 600 * round)) &
    score $P/$name/$order/gpu1/$cell 1 $((27300 + 600 * round)) &
    wait
  done
done
log "queue l4 done"
