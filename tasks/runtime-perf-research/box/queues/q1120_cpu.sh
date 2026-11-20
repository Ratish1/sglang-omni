#!/usr/bin/env bash
# 11-20, node-0000, #3452 review rework (9ff361195) against main (4be177591) and the PR head (734bbb766), CPU only:
# units, byte identity (serial against the video lane), c1 lone video stage time, short requests beside eight videos.
set -u
O=/data/v1120/cpu
K=/workspace/kit/tasks/runtime-perf-research/box
CPUS=0-55,112-167
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> /data/v1120/queue.log; }
log "q1120_cpu start"
cd /workspace/sglang-omni
python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni/test_pipeline.py tests/unit_test/preprocessing/test_video.py \
  tests/unit_test/utils/test_cpu.py > $O/units_touched.log 2>&1
log "units touched rc $? $(tail -1 $O/units_touched.log)"
python3 -m pytest -q -p no:cacheprovider tests/unit_test > $O/units_all.log 2>&1
log "units all rc $? $(tail -1 $O/units_all.log)"
run() {  # arm script args...
  local arm=$1; shift
  PYTHONPATH=/workspace/$arm OMP_NUM_THREADS=112 taskset -c $CPUS python3 "$@"
}
run main $K/preprocess_shared_identity.py --threads 1 --image $O/id.png --out $O/id_main.json > $O/id_main.log 2>&1
log "identity main rc $?"
run vlane $K/preprocess_shared_identity.py --threads 8 --image $O/id.png --out $O/id_vlane.json --against $O/id_main.json > $O/id_vlane.log 2>&1
log "identity vlane rc $? $(grep -c 'differ \[\]' $O/id_vlane.log) clean lines"
for arm in main vlane vlane main; do
  index=$(ls $O | grep -c "^c1_$arm.*json")
  if [ $arm = main ]; then config=1:112:16; else config=8:112:16; fi
  run $arm $K/preprocess_c1_profile.py --configs $config --out $O/c1_${arm}_$index.json > $O/c1_${arm}_$index.log 2>&1
  log "c1 $arm $index rc $? $(tail -1 $O/c1_${arm}_$index.log)"
done
for arm in main vhead vlane vlane vhead main; do
  index=$(ls $O | grep -c "^short_$arm.*json")
  if [ $arm = main ]; then concurrency=1; else concurrency=8; fi
  run $arm $K/audio_beside_video.py --max-concurrency $concurrency --out $O/short_${arm}_$index.json > $O/short_${arm}_$index.log 2>&1
  log "short $arm $index rc $?"
done
log "q1120_cpu done"
