#!/usr/bin/env bash
# After queue_snake_fp32.sh passes: #3097 with the fp32 SnakeBeta against main as #3097's body reads it, SeedTTS EN, all
# 1,088 prompts, streaming: seeded c1 at seeds 1234 and 4321 (codes held equal: SNR, WER per prompt, SIM), c16 and c64
# in both card orders; then the tables, paired latencies and the first audio after the opening wave.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=${M:?the main tree}
PF=${PF:?the fix tree}
L=$D/queue.log
until grep -q queue_snake_fp32_done $L; do sleep 20; done
if sed -n '/queue snake fp32 start/,$p' $L | grep -q "^stopped"; then
  echo "queue snake fp32 serving skipped: the gate queue stopped" >> $L
  echo "queue_snake_fp32_serving_done $(date +%T)" >> $L
  exit 1
else
  :
fi
echo "queue snake fp32 serving start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) fix $(git -C $PF rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $IN/pair.py --label $label --tree $M --b-tree $PF "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair p2fs_c1_1234 --concurrency 1 --a-card 2 --b-card 3 --port 10000 --seed 1234
run_pair p2fs_c1_4321 --concurrency 1 --a-card 3 --b-card 2 --port 11200 --seed 4321
for label in p2fs_c1_1234 p2fs_c1_4321; do
  if [ -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
    :
  else
    python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
    echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
  fi
done
python3 $SEEDED/seeded_quality.py p2fs_c1_1234 p2fs_c1_4321 > $D/seeded_quality_p2f.txt 2>&1
run_pair p2f_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 12400
run_pair p2f_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 13600
run_pair p2f_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 17000
run_pair p2f_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 18200
LABELS="p2fs_c1_1234 p2fs_c1_4321 p2f_c16_a p2f_c16_b p2f_c64_a p2f_c64_b"
python3 $IN/pair_table.py $LABELS > $D/table_p2f.txt 2>&1
python3 $READ/paired_by_prompt.py $LABELS > $D/paired_p2f.txt 2>&1
python3 $READ/first_audio_tail.py p2f_c16_a p2f_c16_b p2f_c64_a p2f_c64_b > $D/first_audio_tail_p2f.txt 2>&1
echo "queue_snake_fp32_serving_done $(date +%T)" >> $L
