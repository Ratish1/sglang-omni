#!/usr/bin/env bash
# After queue_d1_sim.sh: the L1 pairs whose similarity pass ran out of memory on a capped clip (163 s of audio) are
# scored again without the clips past 60 s in either arm (similarity_excluding_caps.py, against each sample's own
# clip), then the codes-held-equal reading and the L1 tables again.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
SHORT=$D/short_reference_meta
M=${M:?the main tree}
L=$D/queue.log
until grep -q queue_d1_sim_done $L; do sleep 20; done
echo "queue d1 sim2 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
for label in d1s96_1234 d1s96_4321 d1_L1_c16_a d1_L1_c16_b d8d1_L1_c16_a d8d1_L1_c16_b; do
  if [ -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
    :
  else
    python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 --meta $SHORT/L1_c16_run.lst > $D/similarity_$label.out 2>&1
    echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
  fi
done
python3 $SEEDED/seeded_quality.py d1s_L1_c1 d1s96_1234 d1s96_4321 > $D/seeded_quality_d1_all.txt 2>&1
python3 $IN/pair_table.py d1s_L1_c1 d1s96_1234 d1s96_4321 d1_L1_c16_a d1_L1_c16_b > $D/table_d1_quality.txt 2>&1
python3 $IN/pair_table.py d8d1_L1_c16_a d8d1_L1_c16_b > $D/table_d8d1_quality.txt 2>&1
echo "queue_d1_sim2_done $(date +%T)" >> $L
