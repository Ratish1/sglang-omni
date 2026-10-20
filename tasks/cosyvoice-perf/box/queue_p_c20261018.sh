#!/usr/bin/env bash
# 10-18, cards 2 and 3: the vocoder's steps by deadline, v2 (first audio due on arrival, started streams when their
# audio runs out, a row passed over once going next, a first audio's step without the streams that can still yield
# and hold more audio than the last step took). Replaces queue_o's v1, whose bound was two of the longest of 16 step
# walls (stopped at 18:06 for this). t_dl2 (perf/cosyvoice3-vocoder-deadline-steps-v2 616ed64db on main 369776998),
# t_stdl2 (prof/cosyvoice3-full-stack-deadline-v2 c1b50ef9a: the full stack t_st with it). Units, warm smoke, the pair
# p_stdl2 (t_st against t_stdl2), the kit profile of t_st and t_stdl2 at the same time (c16), then the pairs p_stdl2m
# (t_ic against t_stdl2) and p_dl2 (t_ic against t_dl2); each pair c1 seeded, c1u, c16, c64, r16, both orders.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_p.pgid; exec bash queue_p_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_p.txt; }
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
units t_dl2 $COSY_SUITES
units t_stdl2 $SUITES
bash $IN/warm_c20261018.sh t_dl2 2 0-15,64-79 26100 $D/warm_dl2 &
bash $IN/warm_c20261018.sh t_stdl2 3 16-31,80-95 26500 $D/warm_stdl2 &
wait
for w in warm_dl2 warm_stdl2; do
  log "warm $w $(grep 'bench rc' $D/$w/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/$w/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/$w/serve.log)"
done

pair p_stdl2 t_st t_stdl2 213a88be5 c1b50ef9a

mkdir -p $D/prof_dl2
PEER=$D/prof_dl2/stdl2 PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_st 2 0-15,64-79 27100 \
  $D/prof_dl2/st > $D/prof_dl2/run_st.log 2>&1 &
PEER=$D/prof_dl2/st PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_stdl2 3 16-31,80-95 27500 \
  $D/prof_dl2/stdl2 > $D/prof_dl2/run_stdl2.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_dl2/st/progress.txt) / $(tail -n 1 $D/prof_dl2/stdl2/progress.txt)"
python3 $IN/first_audio_timeline_c20261017.py st=$D/prof_dl2/st/c16.sqlite stdl2=$D/prof_dl2/stdl2/c16.sqlite \
  > $D/prof_dl2/timeline_c16.txt 2>&1
python3 $IN/chunk_timeline_c20261018.py st=$D/prof_dl2/st/c16.sqlite stdl2=$D/prof_dl2/stdl2/c16.sqlite \
  > $D/prof_dl2/chunks_c16.txt 2>&1
log "analyses"

pair p_stdl2m t_ic t_stdl2 d50266ba6 c1b50ef9a
pair p_dl2 t_ic t_dl2 d50266ba6 616ed64db
log "queue done"
