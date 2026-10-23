#!/usr/bin/env bash
# After queue_d1_seeds.sh: #3097 (D2 with D9) against main on one truth. vocoder_truth_probe.py on each tree (main card
# 2, D2 card 3, at once), the same codes from the L1 c16 list's distinct clips, a SeedTTS-like stream and a long one per
# clip, every decode through the arena as served; then the per-clip SNR to the fp32 decode, D2 minus main.
set -u
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
M=${M:?the main tree}
P2=${P2:?the D2 tree}
L=$D/queue.log
until grep -q queue_d1_seeds_done $L; do sleep 20; done
echo "queue truth start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d2 $(git -C $P2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $M && PYTHONPATH=$M python3 $HERE/vocoder_truth_probe.py run $LONG/L1_c16_run.lst $D/truth_main.json --card 2 > $D/truth_main.txt 2>&1) &
(cd $P2 && PYTHONPATH=$P2 python3 $HERE/vocoder_truth_probe.py run $LONG/L1_c16_run.lst $D/truth_d2.json --card 3 > $D/truth_d2.txt 2>&1) &
wait
echo "truth runs $(tail -n 1 $D/truth_main.txt) / $(tail -n 1 $D/truth_d2.txt) $(date +%T)" >> $L
(cd $M && PYTHONPATH=$M python3 $HERE/vocoder_truth_probe.py compare $D/truth_main.json $D/truth_d2.json > $D/truth_main_d2.txt 2>&1)
echo "queue_truth_done $(date +%T)" >> $L
