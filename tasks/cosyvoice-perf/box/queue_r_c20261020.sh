#!/usr/bin/env bash
# 10-20, card 3 beside queue q's scoring on card 2. The L1 PR branch (perf/cosyvoice3-overlapped-vocoder-steps, tree
# t_l1pr: L1 with fixups folded, the abort check at a step's completion and its test): the new test called directly
# against L1's package (fails) and the branch's (passes), then the touched suites of the branch and of main t_m21.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_r.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_r_c20261020.sh'
D=/data/c20261020
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner tests/unit_test/scheduling
  tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py"
TEST=test_a_stream_whose_abort_lands_while_the_pump_holds_its_state_gets_nothing
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-overlapped-vocoder-steps
[ -d /workspace/t_l1pr ] || git worktree add -q --detach /workspace/t_l1pr FETCH_HEAD
log "tree t_l1pr $(git -C /workspace/t_l1pr log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/t_l1pr rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l1pr diff --quiet HEAD -- sglang_omni || echo DIRTY)"

until grep -q "pair p_c1f rc" $D/queue_q.txt 2> /dev/null; do sleep 30; done
for tree in t_l1 t_l1pr; do
  result=$(cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c 16-31,80-95 \
    python3 - 2>&1 << PY | tail -n 2 | tr '\n' ' '
import importlib.util
import pytest
import sglang_omni
spec = importlib.util.spec_from_file_location("streaming_t_l1pr", "/workspace/t_l1pr/tests/unit_test/fun_cosyvoice3/test_streaming.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(sglang_omni.__file__)
monkeypatch = pytest.MonkeyPatch()
try:
    module.$TEST(monkeypatch)
    print("PASSED")
except AssertionError as error:
    print("FAILED", repr(error)[:120])
finally:
    monkeypatch.undo()
PY
)
  log "direct $TEST package $tree: $result"
done
for tree in t_l1pr t_m21; do
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c 16-31,80-95 \
    python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_r_$tree.log 2>&1)
  log "units $tree rc $? $(grep -E 'passed|failed' $D/units/units_r_$tree.log | tail -n 1)"
done
log "queue done"
