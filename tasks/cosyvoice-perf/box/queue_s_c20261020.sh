#!/usr/bin/env bash
# 10-21, cards 2 and 3. The L1 PR branch revised to SGLang's upload pattern (tmp/l1-pr-sglang-uploads 20a1c7b1c, tree
# t_l1pr2: every host list uploaded pinned when the platform has pinned memory and non blocking, no platform checks,
# the stream's prompt uploaded once at its latch, the sine patch at load): its touched suites on card 3, then the pair
# p_l1s (t_m21 main against t_l1pr2) over c1f, c16, c64 and r16, both orders.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_s.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_s_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner tests/unit_test/scheduling
  tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py"
log() { echo "$* $(date +%T)" >> $D/queue_s.txt; }

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
git fetch -q origin tmp/l1-pr-sglang-uploads
[ -d /workspace/t_l1pr2 ] || git worktree add -q --detach /workspace/t_l1pr2 FETCH_HEAD
log "tree t_l1pr2 $(git -C /workspace/t_l1pr2 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/t_l1pr2 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l1pr2 diff --quiet HEAD -- sglang_omni || echo DIRTY)"
[ -d $D/inductor_t_l1pr2 ] || cp -a $D/inductor_t_l1 $D/inductor_t_l1pr2

(cd /workspace/t_l1pr2 && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/t_l1pr2:$COSYVOICE taskset -c 16-31,80-95 \
  python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_s_t_l1pr2.log 2>&1)
log "units t_l1pr2 rc $? $(grep -E 'passed|failed' $D/units/units_s_t_l1pr2.log | tail -n 1)"

(cd /workspace/sglang-omni && PAIR_CELLS=c1f,c16,c64,r16 python3 $IN/serve_pair_c20261020.py /workspace/t_m21 \
  /workspace/t_l1pr2 $D/p_l1s > $D/p_l1s.log 2>&1)
log "pair p_l1s rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_l1s.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
log "queue done"
