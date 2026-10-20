#!/usr/bin/env bash
# 10-18, cards 2 and 3: the vocoder's steps by deadline (first audio due on arrival, started streams when their audio
# runs out, no row past two recent steps, a first audio's step without the streams holding audio). t_dl
# (perf/cosyvoice3-vocoder-deadline-steps 8c516291e on main 369776998), t_stdl (prof/cosyvoice3-full-stack-deadline
# 875d6da41: the full stack t_st with it). Units, warm smoke, the pair p_stdl (t_st against t_stdl), the kit profile of
# t_st and t_stdl at the same time (c16), then the pairs p_stdlm (t_ic against t_stdl) and p_dl (t_ic against t_dl);
# each pair c1 seeded, c1u, c16, c64, r16, both orders, with its gap breakdown.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_o.pgid; exec bash queue_o_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_o.txt; }
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
COSY_SUITES="tests/unit_test/pipeline/test_scheduler.py tests/unit_test/pipeline/test_stage.py
  tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/fun_cosyvoice3"
SUITES="tests/unit_test/model_runner tests/unit_test/qwen3_omni/test_talker_token_readback.py
  tests/unit_test/qwen3_omni/test_talker_row_ownership.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py tests/unit_test/pipeline/test_async_decode.py $COSY_SUITES"

log "queue start"
units t_dl $COSY_SUITES
units t_stdl $SUITES
bash $IN/warm_c20261018.sh t_dl 2 0-15,64-79 26100 $D/warm_dl &
bash $IN/warm_c20261018.sh t_stdl 3 16-31,80-95 26500 $D/warm_stdl &
wait
for w in warm_dl warm_stdl; do
  log "warm $w $(grep 'bench rc' $D/$w/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/$w/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/$w/serve.log)"
done

pair p_stdl t_st t_stdl 213a88be5 875d6da41

mkdir -p $D/prof_dl
PEER=$D/prof_dl/stdl PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_st 2 0-15,64-79 27100 \
  $D/prof_dl/st > $D/prof_dl/run_st.log 2>&1 &
PEER=$D/prof_dl/st PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_stdl 3 16-31,80-95 27500 \
  $D/prof_dl/stdl > $D/prof_dl/run_stdl.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_dl/st/progress.txt) / $(tail -n 1 $D/prof_dl/stdl/progress.txt)"
python3 $IN/first_audio_timeline_c20261017.py st=$D/prof_dl/st/c16.sqlite stdl=$D/prof_dl/stdl/c16.sqlite \
  > $D/prof_dl/timeline_c16.txt 2>&1
python3 $IN/chunk_timeline_c20261018.py st=$D/prof_dl/st/c16.sqlite stdl=$D/prof_dl/stdl/c16.sqlite \
  > $D/prof_dl/chunks_c16.txt 2>&1
log "analyses"

pair p_stdlm t_ic t_stdl d50266ba6 875d6da41
pair p_dl t_ic t_dl d50266ba6 8c516291e
log "queue done"
