#!/usr/bin/env bash
# The PR pairs left at the 10-22 halt (STATE.md 1), on a fresh lease of the same host, cards 2 and 3, against main
# 3a4e73436; nothing already measured runs again. D2 first (nothing of it ran): seeded c1 at seeds 1234 and 4321 (codes
# held equal), c16 and c64 in both card orders. D8: the cap rate of the two prompts its arms capped, each repeated 100
# times, streaming, c16, unseeded, both card orders, no scoring; one more c16 pair, since neither earlier c16 pair
# scored D8's SIM (a capped clip, then the halt) and their audio went with the lease; c64 in both orders; L1 c16 and
# L3 c16 in both orders; non-streaming seeded c1 and c16. D4: non-streaming c16 in the other card order. Then speaker
# similarity excluding capped clips for every new SeedTTS EN pair with an unscored arm, and the tables.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
CAPS=$SEEDED/cap_count.py
LONG=/data/prof_20261022/formal/meta_w2/long
M=${M:?the main tree}
P4=${P4:?the D4 tree}
P8=${P8:?the D8 tree}
P2=${P2:?the D2 tree}
L=$D/queue.log
BASES="10000 11200 12400 13600 17000 18200 19400 20600 21800 23000 24200 25400 26600 27800"
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
echo "queue prs4 start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d4 $(git -C $P4 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d8 $(git -C $P8 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d2 $(git -C $P2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_prs4_done $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  local label=$1 b=$2
  shift 2
  python3 $IN/pair.py --label $label --tree $M --b-tree $b "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
tables() {
  local prefix=$1 streaming
  local labels=$(ls $PAIR_ROOT | grep "^$prefix" | grep -v "^runaway" | sort)
  streaming=$(echo "$labels" | grep -v "ns_" | grep -v "_L[0-9]")
  python3 $IN/pair_table.py $labels > $D/table_$prefix.txt 2>&1
  python3 $READ/paired_by_prompt.py $(echo "$labels" | grep -v "ns_") > $D/paired_$prefix.txt 2>&1
  python3 $READ/first_audio_tail.py $streaming > $D/first_audio_tail_$prefix.txt 2>&1
}
run_pair p2s_c1_1234 $P2 --concurrency 1 --a-card 2 --b-card 3 --port 10000 --seed 1234
run_pair p2s_c1_4321 $P2 --concurrency 1 --a-card 3 --b-card 2 --port 11200 --seed 4321
python3 $SEEDED/seeded_quality.py p2s_c1_1234 p2s_c1_4321 > $D/seeded_quality_p2.txt 2>&1
run_pair p2_c16_a $P2 --concurrency 16 --a-card 2 --b-card 3 --port 12400
run_pair p2_c16_b $P2 --concurrency 16 --a-card 3 --b-card 2 --port 13600
run_pair p2_c64_a $P2 --concurrency 64 --a-card 2 --b-card 3 --port 17000
run_pair p2_c64_b $P2 --concurrency 64 --a-card 3 --b-card 2 --port 18200
tables p2
echo "d2 done $(date +%T)" >> $L
(cd $P8 && PYTHONPATH=$P8 python3 $HERE/repeat_meta.py $D/runaway_p8 \
  common_voice_en_691043-common_voice_en_691044:100 common_voice_en_17275818-common_voice_en_17275816:100 >> $L 2>&1)
run_pair runaway_p8_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 19400 --meta $D/runaway_p8/meta.lst --skip-score
run_pair runaway_p8_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 20600 --meta $D/runaway_p8/meta.lst --skip-score
python3 $CAPS runaway_p8_c16_a runaway_p8_c16_b > $D/caps_runaway_p8.txt 2>&1
run_pair p8_c16_c $P8 --concurrency 16 --a-card 2 --b-card 3 --port 21800
run_pair p8_c64_a $P8 --concurrency 64 --a-card 2 --b-card 3 --port 23000
run_pair p8_c64_b $P8 --concurrency 64 --a-card 3 --b-card 2 --port 24200
run_pair p8_L1_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 25400 --meta $LONG/L1_c16_run.lst
run_pair p8_L1_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 26600 --meta $LONG/L1_c16_run.lst
run_pair p8_L3_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 27800 --meta $LONG/L3_c16_run.lst
run_pair p8_L3_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 10000 --meta $LONG/L3_c16_run.lst
run_pair p8ns_c1 $P8 --concurrency 1 --a-card 3 --b-card 2 --port 11200 --seed 1234 --no-stream
run_pair p8ns_c16 $P8 --concurrency 16 --a-card 2 --b-card 3 --port 12400 --no-stream
run_pair p4ns_c16_b $P4 --concurrency 16 --a-card 3 --b-card 2 --port 13600 --no-stream
for label in p2s_c1_1234 p2s_c1_4321 p2_c16_a p2_c16_b p2_c64_a p2_c64_b p8_c16_c p8_c64_a p8_c64_b p8ns_c1 p8ns_c16 p4ns_c16_b; do
  if [ ! -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] || [ ! -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
    python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
    echo "similarity excluding caps $label rc $? $(date +%T)" >> $L
  else
    :
  fi
done
tables p2
tables p8
tables p4
echo "queue_prs4_done $(date +%T)" >> $L
