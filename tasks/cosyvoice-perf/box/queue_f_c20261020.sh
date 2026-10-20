#!/usr/bin/env bash
# 10-20, cards 2 and 3. Item 3 v4 on main 0029194c4 as three stacked branches: prerequisite 1, the CUDA speech
# tokenizer on one stream, one run at a time, its arena grown at startup (fix/cosyvoice3-speech-tokenizer-startup-
# arena-v2 eadd8b442, tree t_q1); prerequisite 2, expandable segments for the CUDA vocoder process
# (perf/cosyvoice3-expandable-segments-v2 dca4d8205, tree t_q2); item 3, the engine before the vocoder and the prefix
# pool from the memory left (perf/cosyvoice3-prefix-pool-sized 3824f87e7, tree t_i3). The touched suites of each,
# the long form and worst step lists, warms, the worst step check of t_i3 at 16 and 32 rows, pairs of each part on
# its base (p_q1 main against t_q1, p_q2 t_q1 against t_q2, p_i3 t_q2 against t_i3) and of the stack on main (p_s3),
# WER and SIM of p_s3 c16. Scripts from the trace marks checkout, updated from origin first.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_f.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_f_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
M=t_m20
Q1=t_q1
Q2=t_q2
I3=t_i3
log() { echo "$* $(date +%T)" >> $D/queue_f.txt; }
units() {
  (
    cd /workspace/$1
    CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=/workspace/$1 taskset -c $3 python3 -m pytest -q -p no:cacheprovider \
      tests/unit_test/fun_cosyvoice3 tests/unit_test/test_stage_device_contract.py \
      tests/unit_test/scheduling/test_engine_factory.py > $D/units/units_$1.log 2>&1
    echo "units $1 rc $? $(grep -E 'passed|failed' $D/units/units_$1.log | tail -n 1) $(date +%T)" >> $D/queue_f.txt
  )
}
startup() {
  grep -E 'speech tokenizer|largest admitted step|prefix cache reserve|Flow prefix cache|graphs captured|KV Cache is allocated|PYTORCH' \
    $1 | cut -c1-240 | sed 's/^/    /' >> $D/queue_f.txt
}
score() {
  local dir=$1
  (cd /workspace/$M && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/$M && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}
pair() {
  (cd /workspace/sglang-omni && PAIR_CELLS=$4 python3 $IN/serve_pair_c20261020.py /workspace/$1 /workspace/$2 $D/$3 \
    > $D/$3.log 2>&1)
  log "pair $3 rc $?"
}

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
mkdir -p $D/units
cd /workspace/sglang-omni
for pair in "$Q1 fix/cosyvoice3-speech-tokenizer-startup-arena-v2" "$Q2 perf/cosyvoice3-expandable-segments-v2" \
  "$I3 perf/cosyvoice3-prefix-pool-sized"; do
  set -- $pair
  git fetch -q origin $2
  [ -d /workspace/$1 ] || git worktree add -q --detach /workspace/$1 FETCH_HEAD
done
for tree in $M $Q1 $Q2 $I3; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

units $I3 2 0-15,64-79 &
(units $Q1 3 16-31,80-95; units $Q2 3 16-31,80-95) &
wait

(cd /workspace/$M && PYTHONPATH=/workspace/$M python3 $IN/make_longform_meta.py --out $D/longform > $D/lists.log 2>&1 &&
  PYTHONPATH=/workspace/$M python3 $IN/make_worst_meta_c20261011.py --out $D/worst --requests 16 >> $D/lists.log 2>&1 &&
  PYTHONPATH=/workspace/$M python3 $IN/make_worst_meta_c20261011.py --out $D/worst32 --requests 32 >> $D/lists.log 2>&1)
log "lists rc $? long form $(wc -l < $D/longform/meta.lst) worst $(wc -l < $D/worst/meta.lst) worst32 $(wc -l < $D/worst32/meta.lst)"

bash $IN/warm_c20261020.sh $I3 3 16-31,80-95 26700 $D/warm_$I3 &
bash $IN/warm_c20261020.sh $Q1 2 0-15,64-79 26900 $D/warm_$Q1 &
wait
bash $IN/warm_c20261020.sh $Q2 2 0-15,64-79 27100 $D/warm_$Q2
for tree in $Q1 $Q2 $I3; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
  startup $D/warm_$tree/serve.log
done

python3 $IN/worst_step_c20261020.py /workspace/$I3 3 16-31,80-95 27300 $D/worst_i3_16 16 $D/worst/meta.lst \
  > $D/worst_i3_16.txt 2>&1 &
python3 $IN/worst_step_c20261020.py /workspace/$I3 2 0-15,64-79 27500 $D/worst_i3_32 32 $D/worst32/meta.lst \
  > $D/worst_i3_32.txt 2>&1 &
wait
log "worst step 16 rows $(grep -c . $D/worst_i3_16.txt) lines, 32 rows $(grep -c . $D/worst_i3_32.txt) lines"

pair $M $Q1 p_q1 c1,c1u,c16,c64,r16
pair $Q1 $Q2 p_q2 c1,c1u,c16,c64,r16
pair $Q2 $I3 p_i3 c1,c1u,c16,c64,r16,L16
pair $M $I3 p_s3 c1,c1u,c16,c64,r16,L16
for dir in p_s3/a/gpu0/c16 p_s3/a/gpu1/c16 p_s3/b/gpu0/c16 p_s3/b/gpu1/c16; do
  score $D/$dir
done
log "queue done"
