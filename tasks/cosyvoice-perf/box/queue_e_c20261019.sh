#!/usr/bin/env bash
# 10-19, cards 2 and 3: async decode on the S2 plus fix 1 stack (prof/cosyvoice3-overlap-events-async f848e339b,
# tree t_s2ad: t_s2ev + A1, A2, the lookahead fix, the factory wiring). The touched suites, warm, the pair p_s2ad
# against its landing base t_s2ev and p_s2adm against main t_m19 (c1 seeded, c1u, c16, c64, r16, both orders), then
# t_s2ev (card 2) and t_s2ad (card 3) profiled at the same time (c16, c64) and each request's states.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261019/queue_e.pgid; exec bash queue_e_c20261019.sh'
D=/data/c20261019
IN=$D/in
M=t_m19
E=t_s2ev
A=t_s2ad
log() { echo "$* $(date +%T)" >> $D/queue_e.txt; }

log "queue start"
(
  cd /workspace/$A
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$A taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/scheduling/test_streaming_vocoder.py \
    tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/pipeline/test_async_decode.py \
    tests/unit_test/pipeline/test_scheduler.py tests/unit_test/model_runner/test_sglang_execution.py \
    tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py \
    > $D/s2/units_$A.log 2>&1
  echo "units $A rc $? $(grep -E 'passed|failed' $D/s2/units_$A.log | tail -n 1) $(date +%T)" >> $D/queue_e.txt
)
bash $IN/warm_c20261019.sh $A 2 0-15,64-79 26400 $D/warm_s2ad
log "warm $(grep 'bench rc' $D/warm_s2ad/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_s2ad/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_s2ad/serve.log)"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261019.py /workspace/$E \
  /workspace/$A $D/p_s2ad > $D/p_s2ad.log 2>&1)
log "pair p_s2ad rc $?"
(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261019.py /workspace/$M \
  /workspace/$A $D/p_s2adm > $D/p_s2adm.log 2>&1)
log "pair p_s2adm rc $?"

PEER=$D/prof_states/s2ad_peer PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261019.sh $E 2 \
  0-15,64-79 27600 $D/prof_states/s2ev_peer > $D/prof_states/run_s2ev_peer.log 2>&1 &
PEER=$D/prof_states/s2ev_peer PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261019.sh $A 3 \
  16-31,80-95 27800 $D/prof_states/s2ad_peer > $D/prof_states/run_s2ad_peer.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_states/s2ev_peer/progress.txt) / $(tail -n 1 $D/prof_states/s2ad_peer/progress.txt)"
for cell in c16 c64; do
  python3 $IN/request_states_c20261019.py s2ev=$D/prof_states/s2ev_peer/$cell.sqlite \
    s2ad=$D/prof_states/s2ad_peer/$cell.sqlite > $D/prof_states/states_s2ad_$cell.txt 2>&1
done
log "queue done"
