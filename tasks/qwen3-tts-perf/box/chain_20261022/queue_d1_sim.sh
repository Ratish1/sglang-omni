#!/usr/bin/env bash
# After queue_d1_quality.sh: speaker similarity for the L1 pairs against each sample's own SeedTTS EN clip (the long
# reference repeats it; WavLM runs out of memory on the 88 s references), from short_reference_meta.py's lists. Then
# the codes-held-equal reading over the three seeded pairs and the L1 tables with similarity. L3 is not scored: its
# generated audio runs past 100 s.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
SHORT=$D/short_reference_meta
M=${M:?the main tree}
L=$D/queue.log
until grep -q queue_d1_quality_done $L; do sleep 20; done
echo "queue d1 sim start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $M && PYTHONPATH=$M python3 $T/tasks/qwen3-tts-perf/box/profile_20261020/short_reference_meta.py $LONG $SHORT L1_c1 L1_c16 >> $L 2>&1)
similarity() {
  local label=$1 meta=$2 card=$3 arm
  for arm in a b; do
    (cd $M && PYTHONPATH=$M SGLANG_OMNI_STRICT_PORT=1 python3 -m benchmarks.eval.benchmark_tts_seedtts \
      --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --lang en --skip-gpu-cleanup --output-dir $PAIR_ROOT/$label/$arm/bench \
      --meta $meta --similarity-only --device cuda:$card > $PAIR_ROOT/$label/$arm/score--similarity-short.log 2>&1)
    echo "similarity $label $arm rc $? $(grep -oE 'mean[^,]*' $PAIR_ROOT/$label/$arm/score--similarity-short.log | tail -n 1) $(date +%T)" >> $L
  done
}
similarity d1s_L1_c1 $SHORT/L1_c1_run.lst 2
for label in d1s96_1234 d1s96_4321 d1_L1_c16_a d1_L1_c16_b d8d1_L1_c16_a d8d1_L1_c16_b; do
  similarity $label $SHORT/L1_c16_run.lst 2
done
python3 $SEEDED/seeded_quality.py d1s_L1_c1 d1s96_1234 d1s96_4321 > $D/seeded_quality_d1_all.txt 2>&1
python3 $IN/pair_table.py d1s_L1_c1 d1s96_1234 d1s96_4321 d1_L1_c16_a d1_L1_c16_b > $D/table_d1_quality.txt 2>&1
python3 $IN/pair_table.py d8d1_L1_c16_a d8d1_L1_c16_b > $D/table_d8d1_quality.txt 2>&1
echo "queue_d1_sim_done $(date +%T)" >> $L
