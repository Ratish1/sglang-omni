#!/usr/bin/env bash
# #3176 (D3's first step) at its PR head 29e4bc621 against main 67476823a, the standard SeedTTS EN A/B on the 10-25
# lease: the head served had been a755b41e1 (10-24 lease, the same code but for a comment, a docstring and a local
# name), and its unseeded pairs were unscored. Seeded c1 streaming and non-streaming (byte identity expected on all
# 1,088), c16 and c64 in both card orders and non-streaming c16, every unseeded pair scored for WER and SIM; then the
# tables, paired latencies, the first audio after the opening wave and the CI's quality metrics. Cards 2 and 3.
set -u
D=/data/next1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
M=${M:?the main tree}
P3=${P3:?the D3 tree}
L=$D/queue.log
BASES="4400 7000 8200 10000 11200 12400 13600"
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
echo "queue d3 seedtts start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d3 $(git -C $P3 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY]" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_d3_seedtts_done $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  local label=$1
  shift
  python3 $K/v3_20261020/pair.py --label $label --tree $M --b-tree $P3 "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
similarity() {
  local label
  for label in "$@"; do
    if [ -f $PAIR_ROOT/$label/a/bench/similarity_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/similarity_results.json ]; then
      :
    else
      python3 $K/chain_20261022/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
      echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
    fi
  done
}
run_pair d3h_s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 4400 --seed 1234 --skip-score
run_pair d3h_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 7000
run_pair d3h_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 8200
run_pair d3h_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 10000
run_pair d3h_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 11200
run_pair d3h_ns_c1 --concurrency 1 --a-card 3 --b-card 2 --port 12400 --seed 1234 --no-stream --skip-score
run_pair d3h_ns_c16 --concurrency 16 --a-card 2 --b-card 3 --port 13600 --no-stream
similarity d3h_c16_a d3h_c16_b d3h_c64_a d3h_c64_b d3h_ns_c16
LABELS="d3h_s_c1 d3h_c16_a d3h_c16_b d3h_c64_a d3h_c64_b d3h_ns_c1 d3h_ns_c16"
python3 $K/v3_20261020/pair_table.py $LABELS > $D/table_d3h.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py d3h_s_c1 d3h_c16_a d3h_c16_b d3h_c64_a d3h_c64_b > $D/paired_d3h.txt 2>&1
python3 $K/q3t_20261022/first_audio_tail.py d3h_c16_a d3h_c16_b d3h_c64_a d3h_c64_b > $D/first_audio_tail_d3h.txt 2>&1
(cd $M && PYTHONPATH=$M python3 $K/d1merge_20261024/ci_metrics.py d3h_c16_a d3h_c16_b d3h_c64_a d3h_c64_b d3h_ns_c16 \
  > $D/ci_metrics_d3h.txt 2>&1)
echo "queue_d3_seedtts_done $(date +%T)" >> $L
