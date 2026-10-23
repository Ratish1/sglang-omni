#!/usr/bin/env bash
# After queue_d8_on_d1.sh: D1's quality with the codes held equal, scored for speaker similarity too. The pairs before
# kit 2705d80fc ran the similarity pass without --meta, so it looked the long cells' sample ids up in SeedTTS EN and
# skipped every one; their arms are scored again here against their own lists. Then seeded c1 main against D1 on the
# L1 c16 list (96 prompts, 88 s references) at seeds 1234 and 4321, and the codes-held-equal reading over all three
# seeded pairs; then the L1 c16 pairs' tables with similarity.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
M=${M:?the main tree}
P1=${P1:?the D1 tree}
L=$D/queue.log
until grep -q queue_d8_on_d1_done $L; do sleep 20; done
echo "queue d1 quality start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d1 $(git -C $P1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
similarity() {
  local label=$1 meta=$2 arm
  for arm in a b; do
    (cd $M && PYTHONPATH=$M SGLANG_OMNI_STRICT_PORT=1 python3 -m benchmarks.eval.benchmark_tts_seedtts \
      --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --lang en --skip-gpu-cleanup --output-dir $PAIR_ROOT/$label/$arm/bench \
      --meta $meta --similarity-only --device cuda:2 > $PAIR_ROOT/$label/$arm/score--similarity-meta.log 2>&1)
    echo "similarity $label $arm rc $? $(date +%T)" >> $L
  done
}
similarity d1s_L1_c1 $LONG/L1_c1_run.lst
for run in "d1s96_1234 1234 2 3 24200" "d1s96_4321 4321 3 2 25400"; do
  set -- $run
  python3 $IN/pair.py --label $1 --tree $M --b-tree $P1 --concurrency 1 --a-card $3 --b-card $4 --port $5 \
    --meta $LONG/L1_c16_run.lst --seed $2 > $D/$1.out 2>&1
  rc=$?
  failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log)
  failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)
  echo "$1 rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$1.out)" >> $L
done
python3 $SEEDED/seeded_quality.py d1s_L1_c1 d1s96_1234 d1s96_4321 > $D/seeded_quality_d1_all.txt 2>&1
for label in d1_L1_c16_a d1_L1_c16_b d8d1_L1_c16_a d8d1_L1_c16_b; do
  similarity $label $LONG/L1_c16_run.lst
done
python3 $IN/pair_table.py d1s_L1_c1 d1s96_1234 d1s96_4321 d1_L1_c16_a d1_L1_c16_b > $D/table_d1_quality.txt 2>&1
python3 $IN/pair_table.py d8d1_L1_c16_a d8d1_L1_c16_b > $D/table_d8d1_quality.txt 2>&1
echo "queue_d1_quality_done $(date +%T)" >> $L
