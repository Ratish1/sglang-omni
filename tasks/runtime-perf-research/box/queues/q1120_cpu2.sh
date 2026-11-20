#!/usr/bin/env bash
# 11-20, node-0000: the rework with a video's decode shares on the frame pool (6a5aa0f6b), before the GPU pairs: the
# affected suites, the share mutant, identity against main's serial hashes, the thread census under 8 videos, and
# short requests beside 8 videos m h l l h m.
set -u
O=/data/v1120/cpu2
K=/workspace/kit/tasks/runtime-perf-research/box
CPUS=0-55,112-167
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> /data/v1120/queue.log; }
run() {
  local arm=$1; shift
  PYTHONPATH=/workspace/$arm OMP_NUM_THREADS=112 taskset -c $CPUS python3 "$@"
}
log "q1120_cpu2 start $(git -C /workspace/vlane rev-parse --short HEAD) $(git -C /workspace/vlane rev-parse --short=12 HEAD:sglang_omni)"
cd /workspace/vlane
PYTHONPATH=/workspace/vlane python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni tests/unit_test/preprocessing \
  tests/unit_test/minicpm_o tests/unit_test/ming_omni > $O/units_affected.log 2>&1
log "units affected rc $? $(grep -E '[0-9]+ passed' $O/units_affected.log | tail -1)"
T=/workspace/vmut
git -C $T checkout -q -- . && git -C $T checkout -q 6a5aa0f6b
(cd $T && git apply $K/mutants/vlane_shares_on_new_threads.patch && PYTHONPATH=$T python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/preprocessing/test_video.py::test_a_frame_pool_decodes_every_share_of_a_split_video_on_its_threads) \
  > $O/mutant_shares.log 2>&1
rc=$?
log "mutant vlane_shares_on_new_threads rc $rc ($([ $rc = 0 ] && echo SURVIVED || echo killed))"
git -C $T checkout -q -- .
run vlane $K/preprocess_shared_identity.py --threads 8 --image $O/id.png --out $O/id_vlane.json \
  --against /data/v1120/cpu/id_main.json > $O/id_vlane.log 2>&1
log "identity vlane rc $? $(grep -c 'differ \[\]' $O/id_vlane.log) clean lines"
for arm in vlane vhead; do
  run $arm $K/lane_thread_census.py --max-concurrency 8 > $O/census_$arm.log 2>&1
  log "census $arm rc $? $(grep 'total threads' $O/census_$arm.log)"
done
for arm in main vhead vlane vlane vhead main; do
  index=$(ls $O | grep -c "^short_$arm.*json")
  if [ $arm = main ]; then concurrency=1; else concurrency=8; fi
  run $arm $K/audio_beside_video.py --max-concurrency $concurrency --out $O/short_${arm}_$index.json > $O/short_${arm}_$index.log 2>&1
  log "short $arm $index rc $?"
done
log "q1120_cpu2 done"
