#!/usr/bin/env bash
# After queue_profile.sh: D8 and D3's first step, each against main 67476823a alone, cards 2 and 3, default serving.
# D8 (class C upstream of the codes: the speaker embedding moves every prompt's codes): SeedTTS EN seeded c1, c16 and
# c64 in both card orders, non-streaming seeded c1 and c16, every pair scored for WER and SIM; L1 c16 and L3 c16 in
# both orders with request events and each pair's first-audio chain, L1 scored for SIM against each sample's own clip.
# D3 (class A: copies only): the same SeedTTS EN set unscored, its seeded pairs read for byte identity, and L3 c16 in
# both orders. A branch flagged by queue_gate.sh is skipped. Pairs whose arms both finished are skipped on a rerun.
set -u
D=/data/next1024
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
CHAINDIR=$T/tasks/qwen3-tts-perf/box/chain_20261022
CHAIN=$T/tasks/qwen3-tts-perf/box/profile_20261022/first_audio_chain.py
LONG=/data/d1merge/long
SHORT=/data/d1merge/short_reference_meta
M=${M:?the main tree}
P8=${P8:?the D8 tree}
P3=${P3:?the D3 tree}
L=$D/queue.log
until grep -q queue_profile_done $L; do sleep 30; done
BASES="4400 7000 8200 10000 11200 12400 13600 14800 17000 18200 19400 20600 21800 23000 24200 25400 26600 27800 30000 31200"
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
echo "queue pairs start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d8 $(git -C $P8 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d3 $(git -C $P3 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_pairs_done $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  local label=$1 b_tree=$2
  shift 2
  if [ -f $PAIR_ROOT/$label/a/bench/speed_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/speed_results.json ]; then
    return
  else
    :
  fi
  python3 $IN/pair.py --label $label --tree $M --b-tree $b_tree "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
seedtts_similarity() {
  local label
  for label in "$@"; do
    if [ -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
      :
    else
      python3 $CHAINDIR/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
      echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
    fi
  done
}
if [ -f $D/failed_d8 ]; then
  echo "d8 skipped: failed its gate" >> $L
else
  run_pair d8s_c1 $P8 --concurrency 1 --a-card 2 --b-card 3 --port 4400 --seed 1234
  run_pair d8_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 7000
  run_pair d8_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 8200
  run_pair d8_c64_a $P8 --concurrency 64 --a-card 2 --b-card 3 --port 10000
  run_pair d8_c64_b $P8 --concurrency 64 --a-card 3 --b-card 2 --port 11200
  run_pair d8ns_c1 $P8 --concurrency 1 --a-card 3 --b-card 2 --port 12400 --seed 1234 --no-stream
  run_pair d8ns_c16 $P8 --concurrency 16 --a-card 2 --b-card 3 --port 13600 --no-stream
  D8_SEEDTTS="d8s_c1 d8_c16_a d8_c16_b d8_c64_a d8_c64_b d8ns_c1 d8ns_c16"
  seedtts_similarity $D8_SEEDTTS
  run_pair d8_L1_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 14800 --meta $LONG/L1_c16_run.lst --events
  run_pair d8_L1_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 17000 --meta $LONG/L1_c16_run.lst --events
  run_pair d8_L3_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 18200 --meta $LONG/L3_c16_run.lst --events
  run_pair d8_L3_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 19400 --meta $LONG/L3_c16_run.lst --events
  for label in d8_L1_c16_a d8_L1_c16_b d8_L3_c16_a d8_L3_c16_b; do
    python3 $CHAIN $PAIR_ROOT/$label/a/events $PAIR_ROOT/$label/b/events > $D/chain_$label.txt 2>&1
  done
  for label in d8_L1_c16_a d8_L1_c16_b; do
    python3 $CHAINDIR/similarity_excluding_caps.py $label --tree $M --card 2 --meta $SHORT/L1_c16_run.lst \
      > $D/similarity_$label.out 2>&1
    echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
  done
  python3 $IN/pair_table.py $D8_SEEDTTS > $D/table_d8_seedtts.txt 2>&1
  python3 $IN/pair_table.py d8_L1_c16_a d8_L1_c16_b d8_L3_c16_a d8_L3_c16_b > $D/table_d8_long.txt 2>&1
  python3 $READ/paired_by_prompt.py d8s_c1 d8_c16_a d8_c16_b d8_c64_a d8_c64_b d8_L1_c16_a d8_L1_c16_b d8_L3_c16_a \
    d8_L3_c16_b > $D/paired_d8.txt 2>&1
  python3 $READ/first_audio_tail.py d8_c16_a d8_c16_b d8_c64_a d8_c64_b > $D/first_audio_tail_d8.txt 2>&1
  (cd $M && PYTHONPATH=$M python3 $T/tasks/qwen3-tts-perf/box/d1merge_20261024/ci_metrics.py d8ns_c1 d8ns_c16 \
    > $D/ci_metrics_d8.txt 2>&1)
  echo "d8 done $(date +%T)" >> $L
fi
if [ -f $D/failed_d3 ]; then
  echo "d3 skipped: failed its gate" >> $L
else
  run_pair d3s_c1 $P3 --concurrency 1 --a-card 3 --b-card 2 --port 20600 --seed 1234 --skip-score
  run_pair d3_c16_a $P3 --concurrency 16 --a-card 2 --b-card 3 --port 21800 --skip-score
  run_pair d3_c16_b $P3 --concurrency 16 --a-card 3 --b-card 2 --port 23000 --skip-score
  run_pair d3_c64_a $P3 --concurrency 64 --a-card 2 --b-card 3 --port 24200 --skip-score
  run_pair d3_c64_b $P3 --concurrency 64 --a-card 3 --b-card 2 --port 25400 --skip-score
  run_pair d3ns_c1 $P3 --concurrency 1 --a-card 2 --b-card 3 --port 26600 --seed 1234 --no-stream --skip-score
  run_pair d3ns_c16 $P3 --concurrency 16 --a-card 3 --b-card 2 --port 27800 --no-stream --skip-score
  run_pair d3_L3_c16_a $P3 --concurrency 16 --a-card 2 --b-card 3 --port 30000 --meta $LONG/L3_c16_run.lst --skip-score
  run_pair d3_L3_c16_b $P3 --concurrency 16 --a-card 3 --b-card 2 --port 31200 --meta $LONG/L3_c16_run.lst --skip-score
  D3_LABELS="d3s_c1 d3_c16_a d3_c16_b d3_c64_a d3_c64_b d3ns_c1 d3ns_c16 d3_L3_c16_a d3_L3_c16_b"
  python3 $IN/pair_table.py $D3_LABELS > $D/table_d3.txt 2>&1
  python3 $READ/paired_by_prompt.py d3s_c1 d3_c16_a d3_c16_b d3_c64_a d3_c64_b d3_L3_c16_a d3_L3_c16_b \
    > $D/paired_d3.txt 2>&1
  python3 $READ/first_audio_tail.py d3_c16_a d3_c16_b d3_c64_a d3_c64_b > $D/first_audio_tail_d3.txt 2>&1
  echo "d3 done $(date +%T)" >> $L
fi
echo "queue_pairs_done $(date +%T)" >> $L
