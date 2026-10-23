#!/usr/bin/env bash
# D1's SIM with the codes held equal read -0.032 (95 % -0.058 to -0.007) over 198 prompts, while the truth probe puts
# its first decode as close to the fp32 decode as main's (stream SNR minus main's -0.07 dB, 95 % -0.20 to +0.06, over
# 62 clips). Two more seeded c1 pairs main against D1 on the 96-prompt L1 list (seeds 2468 and 1357, one per card
# order), scored for SIM against each sample's own clip without clips past 60 s, then the reading over all five.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
SHORT=$D/short_reference_meta
M=${M:?the main tree}
P1=${P1:?the D1 tree}
L=$D/queue.log
echo "queue d1 seeds start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d1 $(git -C $P1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
for run in "d1s96_2468 2468 2 3 26600" "d1s96_1357 1357 3 2 27800"; do
  set -- $run
  python3 $IN/pair.py --label $1 --tree $M --b-tree $P1 --concurrency 1 --a-card $3 --b-card $4 --port $5 \
    --meta $LONG/L1_c16_run.lst --seed $2 > $D/$1.out 2>&1
  rc=$?
  failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log)
  failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)
  echo "$1 rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$1.out)" >> $L
  python3 $HERE/similarity_excluding_caps.py $1 --tree $M --card 2 --meta $SHORT/L1_c16_run.lst > $D/similarity_$1.out 2>&1
  echo "similarity excluding caps $1 rc $? $(head -n 1 $D/similarity_$1.out) $(date +%T)" >> $L
done
python3 $SEEDED/seeded_quality.py d1s_L1_c1 d1s96_1234 d1s96_4321 d1s96_2468 d1s96_1357 > $D/seeded_quality_d1_five.txt 2>&1
echo "queue_d1_seeds_done $(date +%T)" >> $L
