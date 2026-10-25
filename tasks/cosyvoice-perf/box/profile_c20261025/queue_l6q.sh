#!/usr/bin/env bash
# 10-25, cards 0 and 1: #3132 with its prefix hop tiers at every whole DiT chunk (2f6c524e1, the review's F1 after
# queue_l6t's profile), on its own pair against its landing base, main ab8193645 (0b9be5746's second parent). The
# head and L5 with it merged (957510bc3) come from $P/l6q.bundle. Steps: units of both heads, one card each; a warm boot
# of each pair arm; pair p_l6q (c1f, c16, c64, b1f, b16, both orders) and its readers; WER and SIM of c16 and c1f.
# Decision rules, written before the run: units pass; the head's boot serves 64 of 64 with 64 prefix tiers (50 to
# 3200 frames) and its prefix graphs measured before the pool; no row worse in both orders past p_aac's c16 A/A band
# (3 to 6 % on tail rows), with c16 and c64 req/s at least p_l6c's gains for the old ladder (+2.3 to +3.5 %); WER and
# SIM inside the spread of p_l6c's.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025/l6q/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_l6q.sh'
D=/data/c20261025
P=$D/l6q
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
CELLS=c1f,c16,c64,b1f,b16
export PAIR_HOME=$D
export PAIR_PORTS=44000,44400
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree() {
  local name=$1 sha=$2
  [ -d /workspace/$name ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-60) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
}
bundled() {
  git -C /workspace/sglang-omni fetch -q $P/l6q.bundle $1
  git -C /workspace/sglang-omni rev-parse FETCH_HEAD
}
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-prefix-graph-ladder-v2
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-prefill-graphs-v3
tree t_l6q $(bundled perf/cosyvoice3-prefix-graph-ladder-v2)
tree t_l5q $(bundled perf/cosyvoice3-prefill-graphs-v3)
tree t_mq $(git -C /workspace/sglang-omni rev-parse 0b9be5746^2)
units() {
  local name=$1 card=$2
  (cd /workspace/$name && CUDA_VISIBLE_DEVICES=$card TORCHINDUCTOR_CACHE_DIR=$P/inductor_units_$name \
    PYTHONPATH=/workspace/$name:$COSYVOICE taskset -c $(cpus $card) \
    timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -rfEs \
    > $P/units_$name.log 2>&1)
  log "units $name rc $? $(grep -E 'passed|failed' $P/units_$name.log | tail -n 1)"
}
units t_l6q 0 &
units t_l5q 1 &
wait
grep -q 'units t_l6q rc 0' $P/queue.txt && grep -q 'units t_l5q rc 0' $P/queue.txt || { log "units failed, stop"; exit 1; }
bash $IN/warm_c20261021.sh t_l6q 0 $(cpus 0) 29900 $P/warm_t_l6q &
bash $IN/warm_c20261021.sh t_mq 1 $(cpus 1) 29600 $P/warm_t_mq &
wait
for name in t_l6q t_mq; do
  log "warm $name $(grep 'bench rc' $P/warm_$name/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm_$name/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm_$name/serve.log)"
  grep -E 'prefix graphs captured|Flow prefix cache' $P/warm_$name/serve.log | sed 's/^.*\] //' | sed -E 's/tiers=\[([0-9]+), ([0-9]+), .*, ([0-9]+)\]/tiers=[\1, \2 .. \3]/' | cut -c1-220 | sed 's/^/    /' >> $P/queue.txt
done
python3 $IN/host_load_c20261022.py $P/p_l6q_load.jsonl &
sampler=$!
(cd /workspace/sglang-omni && PAIR_CELLS=$CELLS python3 $IN/serve_pair_c20261021.py /workspace/t_mq /workspace/t_l6q \
  $P/p_l6q > $P/p_l6q.log 2>&1)
log "pair p_l6q rc $? token cap hits $(grep -E '\| at the token cap \|' $P/p_l6q.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
kill $sampler
python3 $IN/host_load_c20261022.py --summary $P/p_l6q_load.jsonl $P/p_l6q $CELLS > $P/p_l6q_load.txt 2>&1
for cell in c16 c64; do
  python3 $IN/pooled_inter_chunk_c20261021.py $P/p_l6q $cell t_mq t_l6q > $P/p_l6q_pooled_$cell.txt 2>&1
  python3 $IN/gap_positions_c20261022.py $P/p_l6q $cell > $P/p_l6q_positions_$cell.txt 2>&1
done
python3 $IN/profile_c20261022/identity.py $P/p_l6q > $P/identity_p_l6q.txt 2>&1
log "identity p_l6q $(grep -E 'c1f|b1f' $P/identity_p_l6q.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-300)"
python3 $IN/profile_c20261024/latency_tail_by_audio.py $P/p_l6q b16 c16 c64 > $P/p_l6q_tails.txt 2>&1
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_mq && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_mq && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
round=0
for cell in c16 c1f; do
  for order in a b; do
    round=$((round + 1))
    score $P/p_l6q/$order/gpu0/$cell 0 $((34000 + 600 * round)) &
    score $P/p_l6q/$order/gpu1/$cell 1 $((34300 + 600 * round)) &
    wait
  done
done
log "queue l6q done"
