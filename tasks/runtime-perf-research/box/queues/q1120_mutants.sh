#!/usr/bin/env bash
# 11-20, node-0000: each #3452 rework guarantee reverted alone on 653198ebc, and the test that guards it, which must fail.
set -u
O=/data/v1120/mutants
K=/workspace/kit/tasks/runtime-perf-research/box
T=/workspace/vmut
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> /data/v1120/queue.log; }
[ -d $T ] || git -C /workspace/sglang-omni worktree add -q --detach $T 653198ebc
cd $T
git checkout -q 653198ebc && git checkout -q -- .
log "mutants start $(git rev-parse --short HEAD) $(git rev-parse --short=12 HEAD:sglang_omni)"
while read -r patch test; do
  git checkout -q -- .
  git apply $K/mutants/$patch.patch || { log "mutant $patch did not apply"; continue; }
  PYTHONPATH=$T python3 -m pytest -q -p no:cacheprovider "$test" > $O/$patch.log 2>&1
  rc=$?
  log "mutant $patch rc $rc ($([ $rc = 0 ] && echo SURVIVED || echo killed)) $(grep -E 'passed|failed' $O/$patch.log | tail -1)"
done <<'EOF'
vlane_media_on_shared_pool tests/unit_test/qwen3_omni/test_pipeline.py::test_preprocessing_audio_request_decodes_while_a_video_holds_its_media_threads
vlane_no_count_restore tests/unit_test/qwen3_omni/test_pipeline.py::test_preprocessing_video_lane_leaves_later_threads_the_process_intraop_threads
vlane_no_prestart tests/unit_test/qwen3_omni/test_pipeline.py::test_preprocessing_video_lane_leaves_later_threads_the_process_intraop_threads
vlane_no_processor_wait tests/unit_test/qwen3_omni/test_pipeline.py::test_qwen_video_lane_request_fails_after_its_processor_stops_reading_frames
vlane_slices_one_at_a_time tests/unit_test/preprocessing/test_video.py::test_a_frame_pool_resizes_a_lone_video_on_every_thread_into_the_bytes_of_one_resize
vlane_local_video_on_shared_pool tests/unit_test/preprocessing/test_video.py::test_a_requests_local_videos_load_on_its_connectors_threads
EOF
git checkout -q -- .
log "mutants done"
