#!/usr/bin/env bash
# After queue_prs2.sh: D4's non-streaming c16 in the other card order (the first capped the known runaway in D4's
# arm); D8's cap rate on common_voice_en_691043-common_voice_en_691044 (capped in D8's seeded c1 arm, not main's),
# repeated 100 times, streaming, c16, unseeded, both card orders, no scoring; then speaker similarity excluding
# capped clips for every SeedTTS EN pair whose scorer left an arm unscored.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
CAPS=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022/cap_count.py
M=${M:?the main tree}
P4=${P4:?the fixed D4 tree}
P8=${P8:?the fixed D8 tree}
L=$D/queue.log
until grep -q queue_prs2_done $L; do sleep 20; done
echo "queue prs3 start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d4 $(git -C $P4 rev-parse HEAD:sglang_omni) d8 $(git -C $P8 rev-parse HEAD:sglang_omni) kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
python3 $IN/pair.py --label p4ns_c16_b --tree $M --b-tree $P4 --concurrency 16 --a-card 3 --b-card 2 --port 24200 \
  --no-stream > $D/p4ns_c16_b.out 2>&1
echo "p4ns_c16_b rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/p4ns_c16_b/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/p4ns_c16_b/b/serve.log)" >> $L
labels=$(ls $PAIR_ROOT | grep "^p4" | sort)
python3 $IN/pair_table.py $labels > $D/table_p4.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_p4.txt 2>&1
(cd $P8 && PYTHONPATH=$P8 python3 $HERE/repeat_meta.py $D/runaway691043 common_voice_en_691043-common_voice_en_691044:100 >> $L 2>&1)
for order in "a 2 3 25400" "b 3 2 26600"; do
  set -- $order
  python3 $IN/pair.py --label runaway691043_c16_$1 --tree $M --b-tree $P8 --concurrency 16 --a-card $2 --b-card $3 \
    --port $4 --meta $D/runaway691043/meta.lst --skip-score > $D/runaway691043_c16_$1.out 2>&1
  echo "runaway691043_c16_$1 rc $? $(date +%T)" >> $L
done
python3 $CAPS runaway691043_c16_a runaway691043_c16_b > $D/caps_runaway691043.txt 2>&1
for label in $(ls $PAIR_ROOT | grep -E "^p[0-9]" | grep -v "_L[0-9]" | sort); do
  if [ ! -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] || [ ! -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
    python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
    echo "similarity excluding caps $label rc $? $(date +%T)" >> $L
  else
    :
  fi
done
echo "queue_prs3_done $(date +%T)" >> $L
