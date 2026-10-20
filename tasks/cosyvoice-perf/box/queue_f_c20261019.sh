#!/usr/bin/env bash
# 10-19, cards 2 and 3, after queue e: S2's launch waits fixed (13.3 of DESIGN_VOCODER_STANDARD_20261019.md: each
# stream's prompt on the device, the prefix layout from pinned memory, two copy events in turn). Trees: t_s2evc
# (prof/cosyvoice3-overlap-events 17ef35048, S2 with the fixes and fix 1) and t_s2adc
# (prof/cosyvoice3-overlap-events-async 349e8902b, the same with async decode).
# 1. copy_and_graph_waits_c20261019.py alone on card 3 (the staging capacity, graph relaunches).
# 2. The touched suites on both trees (cards 2 and 3); the drive on t_s2evc with every synchronizing call listed and
#    its audio compared with main's bit for bit.
# 3. Warm both; profile both at the same time (c16, c64); each request's states, every CUDA call's duration inside
#    the vocoder's launches, the selection waits.
# 4. Pairs, both orders, c1 seeded, c1u, c16, c64, r16: p_fix (t_s2ev against t_s2evc: the fixes alone on their
#    landing base), p_s2evcm (main against t_s2evc: the PR candidate), p_s2adc (t_s2evc against t_s2adc: async decode).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261019/queue_f.pgid; exec bash queue_f_c20261019.sh'
D=/data/c20261019
IN=$D/in
M=t_m19
E=t_s2ev
C=t_s2evc
A=t_s2adc
P=$D/prof_fix
log() { echo "$* $(date +%T)" >> $D/queue_f.txt; }

log "queue start"
mkdir -p $P $D/s2
CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 $IN/copy_and_graph_waits_c20261019.py > $D/s2/copy_and_graph_waits.txt 2>&1
log "probe rc $? $(grep -c . $D/s2/copy_and_graph_waits.txt) lines"

for pair in "$C 2 0-15,64-79" "$A 3 16-31,80-95"; do
  set -- $pair
  (
    cd /workspace/$1
    CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=/workspace/$1 taskset -c $3 python3 -m pytest -q -p no:cacheprovider \
      tests/unit_test/fun_cosyvoice3 tests/unit_test/scheduling/test_streaming_vocoder.py \
      tests/unit_test/scheduling/test_engine_factory.py tests/unit_test/pipeline/test_async_decode.py \
      tests/unit_test/pipeline/test_scheduler.py tests/unit_test/model_runner/test_sglang_execution.py \
      tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py \
      > $D/s2/units_$1.log 2>&1
    echo "units $1 rc $? $(grep -E 'passed|failed' $D/s2/units_$1.log | tail -n 1) $(date +%T)" >> $D/queue_f.txt
  ) &
done
wait
(
  cd $D/s2
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$C taskset -c 16-31,80-95 python3 -u \
    $IN/s2_vocoder_drive_c20261019.py --check-waits --out $D/s2/drive_$C.npz > $D/s2/drive_$C.log 2>&1
  echo "drive $C rc $? $(grep '^steps' $D/s2/drive_$C.log)" >> $D/queue_f.txt
  python3 $IN/s2_vocoder_drive_c20261019.py --compare $D/s2/drive_$M.npz $D/s2/drive_$C.npz > $D/s2/drive_compare_$C.txt 2>&1
  echo "drive compare $(head -n 1 $D/s2/drive_compare_$C.txt) $(date +%T)" >> $D/queue_f.txt
)

bash $IN/warm_c20261019.sh $C 2 0-15,64-79 26500 $D/warm_$C &
bash $IN/warm_c20261019.sh $A 3 16-31,80-95 26700 $D/warm_$A &
wait
for tree in $C $A; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
done

PEER=$P/$A PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261019.sh $C 2 0-15,64-79 27600 \
  $P/$C > $P/run_$C.log 2>&1 &
PEER=$P/$C PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261019.sh $A 3 16-31,80-95 27800 \
  $P/$A > $P/run_$A.log 2>&1 &
wait
log "profiles $(tail -n 1 $P/$C/progress.txt) / $(tail -n 1 $P/$A/progress.txt)"
for cell in c16 c64; do
  python3 $IN/request_states_c20261019.py old=$D/prof_states/s2ev/$cell.sqlite fixed=$P/$C/$cell.sqlite \
    async=$P/$A/$cell.sqlite > $P/states_$cell.txt 2>&1
  for range in "voc.step causal_window" "voc.step leftover" "voc.complete causal_window" "voc.complete leftover" \
    mr.prefill_embeds loop.admit loop.next_batch mr.forward mr.collect; do
    python3 $IN/range_api_calls_c20261019.py old=$D/prof_states/s2ev/$cell.sqlite fixed=$P/$C/$cell.sqlite \
      async=$P/$A/$cell.sqlite --range "$range"
  done > $P/launch_calls_$cell.txt 2>&1
  for tree in $C $A; do
    python3 $IN/vocoder_selection_c20261019.py $P/$tree/$cell.sqlite > $P/selection_${tree}_$cell.txt 2>&1
  done
done
log "analysis done"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261019.py /workspace/$E \
  /workspace/$C $D/p_fix > $D/p_fix.log 2>&1)
log "pair p_fix rc $?"
(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261019.py /workspace/$M \
  /workspace/$C $D/p_s2evcm > $D/p_s2evcm.log 2>&1)
log "pair p_s2evcm rc $?"
(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261019.py /workspace/$C \
  /workspace/$A $D/p_s2adc > $D/p_s2adc.log 2>&1)
log "pair p_s2adc rc $?"
log "queue done"
