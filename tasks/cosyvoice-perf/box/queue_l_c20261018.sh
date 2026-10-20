#!/usr/bin/env bash
# 10-18, cards 2 and 3: the two event fixes together and async decode on them. t_ev12 (prof/cosyvoice3-step-events
# a86a2cdbc: main 369776998 with fix 1 and fix 2), t_st (prof/cosyvoice3-step-events-async-decode 213a88be5: t_ev12
# with A1, A2, the lookahead read fix and async decode as a tts_engine factory argument, on in the config). Waits for
# queue_k. Units, warm smoke, the pairs p_ev12 (t_ic against t_ev12), p_st (t_ev12 against t_st) and p_stm (t_ic
# against t_st), each c1 seeded, c1u, c16, c64, r16 in both orders with its gap breakdown, the kit profile of t_ev12
# and t_st at the same time (c16), then WER and SIM of p_st's c16 audio, both arms in both orders, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_l.pgid; exec bash queue_l_c20261018.sh'
D=/data/c20261018
IN=$D/in
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_l.txt; }
units() {
  local tree=$1
  shift
  mkdir -p $D/units/$tree
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 -m pytest -q -p no:cacheprovider "$@" \
    > $D/units/$tree/units.log 2>&1)
  log "units $tree rc $? $(grep -E 'passed|failed' $D/units/$tree/units.log | tail -n 1)"
}
score() {
  local dir=$1
  (cd /workspace/t_main4 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 28100 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main4 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
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
until grep -q "queue done" $D/queue_k.txt; do sleep 30; done
log "queue start"
units t_ev12 $SUITES
units t_st $SUITES
bash $IN/warm_c20261018.sh t_ev12 2 0-15,64-79 26100 $D/warm_ev12 &
bash $IN/warm_c20261018.sh t_st 3 16-31,80-95 26500 $D/warm_st &
wait
for w in warm_ev12 warm_st; do
  log "warm $w $(grep 'bench rc' $D/$w/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/$w/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/$w/serve.log)"
done

pair p_ev12 t_ic t_ev12 d50266ba6 a86a2cdbc
pair p_st t_ev12 t_st a86a2cdbc 213a88be5
pair p_stm t_ic t_st d50266ba6 213a88be5

mkdir -p $D/prof_st
PEER=$D/prof_st/st PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_ev12 2 0-15,64-79 27100 \
  $D/prof_st/ev12 > $D/prof_st/run_ev12.log 2>&1 &
PEER=$D/prof_st/ev12 PYSPY=0 CELLS="c16:16:640" bash $IN/cosy_profile_run_c20261018.sh t_st 3 16-31,80-95 27500 \
  $D/prof_st/st > $D/prof_st/run_st.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_st/ev12/progress.txt) / $(tail -n 1 $D/prof_st/st/progress.txt)"
python3 $IN/chunk_timeline_c20261018.py ev12=$D/prof_st/ev12/c16.sqlite st=$D/prof_st/st/c16.sqlite \
  > $D/prof_st/chunks_c16.txt 2>&1
python3 $IN/first_audio_timeline_c20261017.py ev12=$D/prof_st/ev12/c16.sqlite st=$D/prof_st/st/c16.sqlite \
  > $D/prof_st/timeline_c16.txt 2>&1
python3 $IN/call_overlap_c20261018.py ev12=$D/prof_st/ev12/c16.sqlite st=$D/prof_st/st/c16.sqlite \
  > $D/prof_st/overlap_c16.txt 2>&1
log "analyses"

for order in a b; do
  for gpu in gpu0 gpu1; do
    score $D/p_st/$order/$gpu/c16
  done
done
log "queue done"
