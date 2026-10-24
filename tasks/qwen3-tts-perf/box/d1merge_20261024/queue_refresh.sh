#!/usr/bin/env bash
# #3102 merged with main after #3097 (11c131654) against main 6e75cba89, every number of its body again on the new
# base, cards 2 and 3. First the non-streaming SeedTTS EN checks behind #3102's failed non-streaming CI WER (3.60 %
# against 1.46 %): seeded c1 (byte identity expected on all 1,088: non-streaming decodes through the decoder's own
# forward) and c16 five times with the card orders alternating, read as the CI reads them (ci_metrics.py: corpus WER,
# cap hits, SIM over the first 50 samples). Pairs whose arms are both scored are skipped. Then the long lists and the first decode's distance to the fp32 decode
# (first_chunk_truth_probe.py, crossed, skipped when its output exists); SeedTTS EN seeded c1, c16 and c64 in both
# orders; L1 c16 and L3 c16 in both orders with request events and each pair's first-audio chain; seeded c1 on the L1
# c1 list and on the 96-prompt L1 list at four seeds, each scored for SIM against its sample's own clip, then the
# codes-held-equal reading over all five; then the tables.
set -u
D=/data/d1merge
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
CHAINDIR=$T/tasks/qwen3-tts-perf/box/chain_20261022
PROFILE=$T/tasks/qwen3-tts-perf/box/profile_20261020
CHAIN=$T/tasks/qwen3-tts-perf/box/profile_20261022/first_audio_chain.py
LONG=$D/long
SHORT=$D/short_reference_meta
M=${M:?the main tree}
P1=${P1:?the 3102 tree}
L=$D/queue.log
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
echo "queue refresh start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')3102 $(git -C $P1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_refresh_done $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  local label=$1
  shift
  if [ -f $PAIR_ROOT/$label/a/bench/wer_results.json ] && [ -f $PAIR_ROOT/$label/b/bench/wer_results.json ]; then
    return
  else
    :
  fi
  python3 $IN/pair.py --label $label --tree $M --b-tree $P1 "$@" > $D/$label.out 2>&1
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
short_similarity() {
  local label=$1 meta=$2
  python3 $CHAINDIR/similarity_excluding_caps.py $label --tree $M --card 2 --meta $meta > $D/similarity_$label.out 2>&1
  echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
}
run_pair rp_ns_c1 --concurrency 1 --a-card 2 --b-card 3 --port 10000 --seed 1234 --no-stream
run_pair rp_ns_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 11200 --no-stream
run_pair rp_ns_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 12400 --no-stream
run_pair rp_ns_c16_c --concurrency 16 --a-card 2 --b-card 3 --port 4400 --no-stream
run_pair rp_ns_c16_d --concurrency 16 --a-card 3 --b-card 2 --port 7000 --no-stream
run_pair rp_ns_c16_e --concurrency 16 --a-card 2 --b-card 3 --port 8200 --no-stream
NONSTREAM_LABELS="rp_ns_c1 rp_ns_c16_a rp_ns_c16_b rp_ns_c16_c rp_ns_c16_d rp_ns_c16_e"
seedtts_similarity $NONSTREAM_LABELS
python3 $IN/pair_table.py $NONSTREAM_LABELS > $D/table_nonstream.txt 2>&1
(cd $M && PYTHONPATH=$M python3 $T/tasks/qwen3-tts-perf/box/d1merge_20261024/ci_metrics.py $NONSTREAM_LABELS > $D/ci_metrics_nonstream.txt 2>&1)
echo "non-streaming done $(date +%T)" >> $L
if [ -f $LONG/L1_c16_run.lst ] && [ -f $SHORT/L1_c16_run.lst ]; then
  :
else
  (cd $M && PYTHONPATH=$M python3 $PROFILE/long_cells_meta.py $LONG >> $L 2>&1)
  (cd $M && PYTHONPATH=$M python3 $PROFILE/short_reference_meta.py $LONG $SHORT L1_c1 L1_c16 >> $L 2>&1)
  echo "lists rc $? $(wc -l < $LONG/L1_c16_run.lst) L1 rows $(date +%T)" >> $L
fi
if grep -q "minus main" $D/truth_probe_crossed.txt 2>/dev/null; then
  :
else
  (cd $P1 && PYTHONPATH=$P1 timeout 3600 python3 $CHAINDIR/first_chunk_truth_probe.py $LONG/L1_c16_run.lst --card 2 \
    --references 96 > $D/truth_probe_crossed.txt 2>&1)
  echo "truth probe rc $? $(grep -c ' frames, start ' $D/truth_probe_crossed.txt) clips $(date +%T)" >> $L
fi
run_pair rp_s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 13600 --seed 1234
run_pair rp_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 14800
run_pair rp_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 17000
run_pair rp_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 18200
run_pair rp_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 19400
STREAM_LABELS="rp_s_c1 rp_c16_a rp_c16_b rp_c64_a rp_c64_b"
seedtts_similarity $STREAM_LABELS
python3 $IN/pair_table.py $STREAM_LABELS $NONSTREAM_LABELS > $D/table_seedtts.txt 2>&1
python3 $READ/paired_by_prompt.py $STREAM_LABELS > $D/paired_seedtts.txt 2>&1
python3 $READ/first_audio_tail.py rp_c16_a rp_c16_b rp_c64_a rp_c64_b > $D/first_audio_tail_seedtts.txt 2>&1
echo "seedtts done $(date +%T)" >> $L
run_pair r_L1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 20600 --meta $LONG/L1_c16_run.lst --events
run_pair r_L1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 21800 --meta $LONG/L1_c16_run.lst --events
run_pair r_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 23000 --meta $LONG/L3_c16_run.lst --events
run_pair r_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 24200 --meta $LONG/L3_c16_run.lst --events
for label in r_L1_c16_a r_L1_c16_b r_L3_c16_a r_L3_c16_b; do
  python3 $CHAIN $PAIR_ROOT/$label/a/events $PAIR_ROOT/$label/b/events > $D/chain_$label.txt 2>&1
done
short_similarity r_L1_c16_a $SHORT/L1_c16_run.lst
short_similarity r_L1_c16_b $SHORT/L1_c16_run.lst
run_pair rs_L1_c1 --concurrency 1 --a-card 2 --b-card 3 --port 25400 --seed 1234 --meta $LONG/L1_c1_run.lst
short_similarity rs_L1_c1 $SHORT/L1_c1_run.lst
for run in "rs96_1234 1234 2 3 26600" "rs96_4321 4321 3 2 27800" "rs96_2468 2468 2 3 30000" "rs96_1357 1357 3 2 31200"; do
  set -- $run
  run_pair $1 --concurrency 1 --a-card $3 --b-card $4 --port $5 --seed $2 --meta $LONG/L1_c16_run.lst
  short_similarity $1 $SHORT/L1_c16_run.lst
done
python3 $SEEDED/seeded_quality.py rs_L1_c1 rs96_1234 rs96_4321 rs96_2468 rs96_1357 > $D/seeded_quality_five.txt 2>&1
LONG_LABELS="rs_L1_c1 rs96_1234 r_L1_c16_a r_L1_c16_b r_L3_c16_a r_L3_c16_b"
python3 $IN/pair_table.py $LONG_LABELS > $D/table_long.txt 2>&1
python3 $READ/paired_by_prompt.py $LONG_LABELS > $D/paired_long.txt 2>&1
echo "queue_refresh_done $(date +%T)" >> $L
