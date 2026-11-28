#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_n: DEV4 (818e48b43, source tree 67ed98c669b2 as dev3; its tests rebuild
# the thread processors where they replace the processor), every Qwen3-Omni suite the dev branch reaches, whole.
set -u
O=/data/v1128b
D4=/workspace/dev4
DEV4_COMMIT=818e48b43
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1128d_n done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_o start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-preprocessing-request-threads
git worktree add -q --detach $D4 $DEV4_COMMIT
log "tree dev4 $(git -C $D4 rev-parse --short HEAD) $(git -C $D4 rev-parse --short=12 HEAD:sglang_omni)"
SUITES="tests/unit_test/qwen3_omni/test_pipeline.py tests/unit_test/qwen3_omni/test_preprocessor.py tests/unit_test/qwen3_omni/test_audio_encoder_batch_dedup.py tests/unit_test/qwen3_omni/test_session_config.py tests/unit_test/qwen3_omni/test_sglang_ar_budget.py"
(cd $D4 && python3 -m pytest -q -p no:cacheprovider $SUITES > $O/unit_dev4.log 2>&1)
log "unit dev4 rc $? $(tail -1 $O/unit_dev4.log)"
log "q1128d_o done"
