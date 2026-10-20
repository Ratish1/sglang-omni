#!/usr/bin/env bash
# V3's PR head (perf/qwen3-tts-vocoder-transformer-fused-kernels 0e573ef55, one commit on upstream main 4809648d4)
# against the V3 head the pairs measured (6f6039ba6): the PR calls the rope kernel directly, gates on the CUDA platform,
# rewords two notes and shares the position, mask and retention helpers between the plain and fused steps, so its
# audio must be byte identical. The touched units (the plain step's tests included), then a seeded c1 pair. Waits for
# queue_landing.sh.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
V3=/workspace/v3_6f6039ba6
PR=/workspace/v3pr_0e573ef55
L=$D/queue.log
until grep -q queue_landing_done $L; do sleep 30; done
echo "queue v3pr start $(date +%T) v3 $(git -C $V3 rev-parse HEAD:sglang_omni) pr $(git -C $PR rev-parse HEAD:sglang_omni)" >> $L
(cd $PR && PYTHONPATH=$PR timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfE > $D/units/v3pr.log 2>&1)
echo "units v3pr rc $? $(grep -E 'passed|failed' $D/units/v3pr.log | tail -n 1) $(date +%T)" >> $L
python3 $IN/pair.py --label v3pr_c1_seed --tree $V3 --b-tree $PR --concurrency 1 --a-card 0 --b-card 1 --port 7000 \
  --seed 1234 > $D/v3pr_c1_seed.out 2>&1
echo "v3pr_c1_seed rc $? $(date +%T) $(grep -h 'byte identical' $D/v3pr_c1_seed.out)" >> $L
echo "queue_v3pr_done $(date +%T)" >> $L
