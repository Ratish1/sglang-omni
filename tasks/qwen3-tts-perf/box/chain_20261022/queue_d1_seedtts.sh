#!/usr/bin/env bash
# #3102 (D1) against main on SeedTTS EN, the usual A/B: D1's references reach no SeedTTS prompt (at most 111 frames
# against a 580-frame reach), but every streaming first plan now computes the reach and every server captures one more
# tail width. Seeded c1 (byte identity expected on all 1,088), c16 and c64 in both card orders, non-streaming seeded c1
# and c16; then the tables, paired latencies and the first audio after the opening wave.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=${M:?the main tree}
P1=${P1:?the D1 tree}
L=$D/queue.log
BASES="10000 11200 12400 13600 17000 18200 19400"
BUSY=$(python3 - $BASES <<'EOF'
import sys
listening = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(table).read().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            listening.add(int(fields[1].rsplit(":", 1)[1], 16))
print(" ".join(str(port) for port in sorted(listening) if any(int(base) <= port <= int(base) + 1100 for base in sys.argv[1:])))
EOF
)
echo "queue d1 seedtts start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d1 $(git -C $P1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_d1_seedtts_done $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  local label=$1
  shift
  python3 $IN/pair.py --label $label --tree $M --b-tree $P1 "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair p1s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 10000 --seed 1234
run_pair p1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 11200
run_pair p1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 12400
run_pair p1_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 13600
run_pair p1_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 17000
run_pair p1ns_c1 --concurrency 1 --a-card 3 --b-card 2 --port 18200 --seed 1234 --no-stream
run_pair p1ns_c16 --concurrency 16 --a-card 2 --b-card 3 --port 19400 --no-stream
for label in p1s_c1 p1_c16_a p1_c16_b p1_c64_a p1_c64_b p1ns_c1 p1ns_c16; do
  if [ -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
    :
  else
    python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
    echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
  fi
done
LABELS="p1s_c1 p1_c16_a p1_c16_b p1_c64_a p1_c64_b p1ns_c1 p1ns_c16"
python3 $IN/pair_table.py $LABELS > $D/table_p1.txt 2>&1
python3 $READ/paired_by_prompt.py p1s_c1 p1_c16_a p1_c16_b p1_c64_a p1_c64_b > $D/paired_p1.txt 2>&1
python3 $READ/first_audio_tail.py p1_c16_a p1_c16_b p1_c64_a p1_c64_b > $D/first_audio_tail_p1.txt 2>&1
echo "queue_d1_seedtts_done $(date +%T)" >> $L
