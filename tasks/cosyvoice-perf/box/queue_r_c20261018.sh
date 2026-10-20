#!/usr/bin/env bash
# 10-18, cards 2 and 3: the PR candidates measured as they would land. t_ship (prof/cosyvoice3-ship-stack fd3dc7b4b:
# main 369776998 with fix 1, async decode (A1, A2, the lookahead fix, the factory wiring) and the deadline steps v2,
# without fix 2, which added nothing on fix 1) and t_evdl (prof/cosyvoice3-events-deadline f7e404c4a: main with fix 1
# and the deadline steps v2, no async decode). Waits for queue_q. Units, warm smoke, then the pairs p_ship (t_ic
# against t_ship) and p_evdl (t_ic against t_evdl), each c1 seeded, c1u, c16, c64, r16, both orders.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_r.pgid; exec bash queue_r_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }
units() {
  local tree=$1
  shift
  mkdir -p $D/units/$tree
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider "$@" \
    > $D/units/$tree/units.log 2>&1)
  log "units $tree rc $? $(grep -E 'passed|failed' $D/units/$tree/units.log | tail -n 1)"
}
pair() {
  local name=$1 first=$2 second=$3 first_head=$4 second_head=$5
  (cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261018.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $?"
  python3 $IN/inter_chunk_mechanism_c20261018.py $D/$name $first_head $second_head c16 c64 r16 \
    > $D/${name}_mechanism.txt 2>&1
}
SUITES="tests/unit_test/model_runner tests/unit_test/qwen3_omni/test_talker_token_readback.py
  tests/unit_test/qwen3_omni/test_talker_row_ownership.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py tests/unit_test/pipeline/test_async_decode.py
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3"

log "queue wait"
until grep -q "queue done" $D/queue_q.txt; do sleep 30; done
log "queue start"
units t_ship $SUITES
units t_evdl $SUITES
bash $IN/warm_c20261018.sh t_ship 2 0-15,64-79 26100 $D/warm_ship &
bash $IN/warm_c20261018.sh t_evdl 3 16-31,80-95 26500 $D/warm_evdl &
wait
for w in warm_ship warm_evdl; do
  log "warm $w $(grep 'bench rc' $D/$w/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/$w/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/$w/serve.log)"
done
pair p_ship t_ic t_ship d50266ba6 fd3dc7b4b
pair p_evdl t_ic t_evdl d50266ba6 f7e404c4a
log "queue done"
