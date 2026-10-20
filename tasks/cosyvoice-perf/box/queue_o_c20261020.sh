#!/usr/bin/env bash
# 10-20, cards 2 and 3. L3 with its copies through pinned memory on the graphs' stream (f8e6f9558, tree t_l3e): in
# prof_l each call's codes came back by a pageable copy that waited for the replay inside the call (165 calls, 10.4 ms
# each), and 171 of the AR's and the vocoder's 252 CUDA calls over 1 ms ended with one of the tokenizer threads' calls.
# The stack rebased on it: t_l6e 9068c2868. Units and the warm of t_l3e on card 3 while queue n scores on card 2, then
# pairs p_l3e (t_l2 against t_l3e) and p_alle (t_m21 against t_l6e), then L2 and L3e profiled at c16 at the same time
# (prof_o) and tokenizer_contention_c20261020.py on both traces.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_o.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_o_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
R=$D/prof_o
log() { echo "$* $(date +%T)" >> $D/queue_o.txt; }
pair() {
  (cd /workspace/sglang-omni && PAIR_CELLS=$4 python3 $IN/serve_pair_c20261020.py /workspace/$1 /workspace/$2 $D/$3 \
    > $D/$3.log 2>&1)
  log "pair $3 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$3.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}
warm() {
  [ -d $D/inductor_$1 ] || cp -a $D/inductor_$2 $D/inductor_$1
  bash $IN/warm_c20261020.sh $1 $3 $4 $5 $D/warm_$1
  log "warm $1 $(grep 'bench rc' $D/warm_$1/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$1/bench.log | tr -s ' ' | tr '\n' ' ') token cap hits $(python3 -c "import json; print(json.load(open('$D/warm_$1/bench/speed_results.json'))['summary']['max_token_hits'])" 2>&1 | tail -n 1) tracebacks $(grep -c Traceback $D/warm_$1/serve.log)"
}

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
for pair in "t_l3e perf/cosyvoice3-speech-tokenizer-graphs" "t_l6e perf/cosyvoice3-prefix-graph-ladder"; do
  set -- $pair
  git fetch -q origin $2
  [ -d /workspace/$1 ] || git worktree add -q --detach /workspace/$1 FETCH_HEAD
  log "tree $1 $(git -C /workspace/$1 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$1 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$1 diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

until grep -q "pair p_alld" $D/queue_n.txt 2> /dev/null; do sleep 30; done
(cd /workspace/t_l3e && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/t_l3e taskset -c 16-31,80-95 python3 -m pytest -q \
  -p no:cacheprovider tests/unit_test/fun_cosyvoice3 > $D/units/units_t_l3e.log 2>&1)
log "units t_l3e rc $? $(grep -E 'passed|failed' $D/units/units_t_l3e.log | tail -n 1)"
warm t_l3e t_l3d 3 16-31,80-95 36800

until grep -q "queue done" $D/queue_n.txt 2> /dev/null; do sleep 30; done
warm t_l6e t_l6d 2 0-15,64-79 37200

pair t_l2 t_l3e p_l3e c1,c1u,c16,c64,r16
pair t_m21 t_l6e p_alle c1,c1u,c16,c64,r16,L16

mkdir -p $R
PEER=$R/t_l3e PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh t_l2 2 0-15,64-79 37600 $R/t_l2 \
  > $R/run_t_l2.log 2>&1 &
PEER=$R/t_l2 PYSPY=0 CELLS="c16:16:320" bash $IN/cosy_profile_run_c20261020.sh t_l3e 3 16-31,80-95 38000 $R/t_l3e \
  > $R/run_t_l3e.log 2>&1 &
wait
log "profiles $(tail -n 1 $R/t_l2/progress.txt) / $(tail -n 1 $R/t_l3e/progress.txt)"
python3 $IN/tokenizer_contention_c20261020.py l2=$R/t_l2/c16.sqlite l3e=$R/t_l3e/c16.sqlite > $R/contention.txt 2>&1
cat $R/contention.txt >> $D/queue_o.txt
log "queue done"
