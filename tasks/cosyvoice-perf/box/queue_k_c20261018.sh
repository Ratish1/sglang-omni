#!/usr/bin/env bash
# 10-18, cards 2 and 3: the two event fixes, each alone on main (t_ic: main 369776998's tree). t_ev1
# (perf/model-runner-reuse-step-events 6acb2e026: the staged token ids and each launch's completion through two reused
# events), t_ev2 (perf/cosyvoice3-hift-pinned-samples 0f326d41f: HiFT's step samples through a pinned buffer and an
# event), t_la2 (perf/cosyvoice3-ar-lookahead-v2 1f12afa5e: A1 and A2 with the lookahead read fixed). Waits for
# queue_j. Units (touched suites, main's result beside each), warm smoke, the kit profile of t_ev1 and t_ev2 at the same
# time (c16, against prof_ic's t_ic) with the AR's event creation and what covers it, then the pairs p_ev1 and p_ev2
# against t_ic (c1 seeded, c1u, c16, c64, r16, both orders) with their gap breakdowns.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_k.pgid; exec bash queue_k_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_k.txt; }
units() {
  local tree=$1 name=$2
  shift 2
  mkdir -p $D/units/$tree
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider "$@" \
    > $D/units/$tree/$name.log 2>&1)
  log "units $tree $name rc $? $(grep -E 'passed|failed' $D/units/$tree/$name.log | tail -n 1)"
}
EV1_SUITES="tests/unit_test/model_runner tests/unit_test/qwen3_omni/test_talker_token_readback.py
  tests/unit_test/qwen3_omni/test_talker_row_ownership.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py tests/unit_test/pipeline/test_async_decode.py
  tests/unit_test/pipeline/test_scheduler.py tests/unit_test/fun_cosyvoice3"
COSY_SUITES="tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3"

log "queue wait"
until grep -q "queue done" $D/queue_j.txt; do sleep 30; done
log "queue start"
units t_ic ev1_suites $EV1_SUITES
units t_ev1 ev1_suites $EV1_SUITES
units t_ev2 cosy_suites $COSY_SUITES
units t_la2 cosy_suites $COSY_SUITES
bash $IN/warm_c20261018.sh t_ev1 2 0-15,64-79 26100 $D/warm_ev1 &
bash $IN/warm_c20261018.sh t_ev2 3 16-31,80-95 26500 $D/warm_ev2 &
wait
for w in warm_ev1 warm_ev2; do
  log "warm $w $(grep 'bench rc' $D/$w/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/$w/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/$w/serve.log)"
done

mkdir -p $D/prof_ev
PEER=$D/prof_ev/ev2 PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_ev1 2 0-15,64-79 27100 \
  $D/prof_ev/ev1 > $D/prof_ev/run_ev1.log 2>&1 &
PEER=$D/prof_ev/ev1 PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_ev2 3 16-31,80-95 27500 \
  $D/prof_ev/ev2 > $D/prof_ev/run_ev2.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_ev/ev1/progress.txt) / $(tail -n 1 $D/prof_ev/ev2/progress.txt)"
python3 $IN/ar_collect_calls_c20261018.py ic=$D/prof_ic/ic/c16.sqlite ev1=$D/prof_ev/ev1/c16.sqlite \
  ev2=$D/prof_ev/ev2/c16.sqlite > $D/prof_ev/collect_c16.txt 2>&1
python3 $IN/call_overlap_c20261018.py ic=$D/prof_ic/ic/c16.sqlite ev1=$D/prof_ev/ev1/c16.sqlite \
  ev2=$D/prof_ev/ev2/c16.sqlite > $D/prof_ev/overlap_c16.txt 2>&1
python3 $IN/chunk_timeline_c20261018.py ic=$D/prof_ic/ic/c16.sqlite ev1=$D/prof_ev/ev1/c16.sqlite \
  ev2=$D/prof_ev/ev2/c16.sqlite > $D/prof_ev/chunks_c16.txt 2>&1
log "analyses"

for arm in ev1 ev2; do
  (cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261018.py /workspace/t_ic \
    /workspace/t_$arm $D/p_$arm > $D/p_$arm.log 2>&1)
  log "pair p_$arm rc $?"
done
python3 $IN/inter_chunk_mechanism_c20261018.py $D/p_ev1 d50266ba6 6acb2e026 c16 c64 r16 > $D/p_ev1_mechanism.txt 2>&1
python3 $IN/inter_chunk_mechanism_c20261018.py $D/p_ev2 d50266ba6 0f326d41f c16 c64 r16 > $D/p_ev2_mechanism.txt 2>&1
log "queue done"
