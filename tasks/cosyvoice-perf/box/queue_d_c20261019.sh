#!/usr/bin/env bash
# 10-19, cards 2 and 3. Part 1: main (t_m19, card 2) and S2 (t_s2b, card 3) profiled at the same time with the
# probe's new per-request marks (sched.admit, sched.prefill), c16 and c64, then each request's states
# (request_states_c20261019.py). Part 2: S2 with fix 1 (prof/cosyvoice3-overlap-events d24d4030a, tree t_s2ev): the
# touched suites, warm, the pair p_s2ev against main (c1 seeded, c1u, c16, c64, r16, both orders), its profile
# (c16, c64) and its states.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261019/queue_d.pgid; exec bash queue_d_c20261019.sh'
D=/data/c20261019
IN=$D/in
M=t_m19
S=t_s2b
E=t_s2ev
log() { echo "$* $(date +%T)" >> $D/queue_d.txt; }

log "queue start"
mkdir -p $D/prof_states
PEER=$D/prof_states/s2 PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261019.sh $M 2 \
  0-15,64-79 27300 $D/prof_states/main > $D/prof_states/run_main.log 2>&1 &
PEER=$D/prof_states/main PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261019.sh $S 3 \
  16-31,80-95 27400 $D/prof_states/s2 > $D/prof_states/run_s2.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_states/main/progress.txt) / $(tail -n 1 $D/prof_states/s2/progress.txt)"
for cell in c16 c64; do
  python3 $IN/request_states_c20261019.py main=$D/prof_states/main/$cell.sqlite s2=$D/prof_states/s2/$cell.sqlite \
    > $D/prof_states/states_$cell.txt 2>&1
done
log "states c16 c64"

(
  cd /workspace/$E
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$E taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/scheduling/test_streaming_vocoder.py \
    tests/unit_test/model_runner/test_sglang_execution.py tests/unit_test/qwen3_omni/test_talker_token_readback.py \
    tests/unit_test/qwen3_tts/test_decode_step_staging.py > $D/s2/units_$E.log 2>&1
  echo "units $E rc $? $(grep -E 'passed|failed' $D/s2/units_$E.log | tail -n 1) $(date +%T)" >> $D/queue_d.txt
)
bash $IN/warm_c20261019.sh $E 2 0-15,64-79 26300 $D/warm_s2ev
log "warm $(grep 'bench rc' $D/warm_s2ev/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_s2ev/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_s2ev/serve.log)"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261019.py /workspace/$M \
  /workspace/$E $D/p_s2ev > $D/p_s2ev.log 2>&1)
log "pair p_s2ev rc $?"

PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261019.sh $E 2 0-15,64-79 27500 \
  $D/prof_states/s2ev > $D/prof_states/run_s2ev.log 2>&1
for cell in c16 c64; do
  python3 $IN/request_states_c20261019.py s2ev=$D/prof_states/s2ev/$cell.sqlite > $D/prof_states/states_s2ev_$cell.txt 2>&1
done
log "queue done"
