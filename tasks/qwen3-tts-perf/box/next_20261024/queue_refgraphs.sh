#!/usr/bin/env bash
# CUDA graphs for the reference encoder past 20 s, read in serving rather than in an idle bench: under L1 c16 load the
# eager encode past the graphs took 29.5 ms for 12.0 ms of device (468 launches, the rest CUDA calls, lock waits and
# Python; profile_20261024 4), where an idle bench saw a graph save about 1 ms for about 1 GB. Arms: D8 version 3
# (b17546a65) against check/d8-reference-graphs (c355b26d2: the reference encoder's ladder extended by 512, 768, 1024
# and 1280 frames, D8's rounding widths, the speaker graphs kept on the default ladder). SeedTTS EN seeded c1 (byte
# identity expected: no SeedTTS reference reaches the new widths); L1 c16 and L3 c16 in both card orders with request
# events and each pair's first-audio chain, unscored; each boot's capture lines, KV pool and startup. Cards 2 and 3.
set -u
D=/data/next1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
CHAIN=$K/profile_20261022/first_audio_chain.py
LONG=$D/long
A=${A:?the D8 version 3 tree}
B=${B:?the check tree}
L=$D/queue.log
echo "queue refgraphs start $(date +%T) d8v3 $(git -C $A rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')check $(git -C $B rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $K/v3_20261020/pair.py --label $label --tree $A --b-tree $B "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair rg_s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 4400 --seed 1234 --skip-score
for arm in a b; do
  echo "rg_s_c1 $arm: $(grep -oE 'KV Cache is allocated[^#]*#tokens: [0-9]+' $PAIR_ROOT/rg_s_c1/$arm/serve.log | tail -1) | $(grep -iE 'reference encoder graph|reference_encoder_graph' $PAIR_ROOT/rg_s_c1/$arm/serve.log | tail -2 | tr '\n' ' ')" >> $L
done
run_pair rg_L1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 7000 --meta $LONG/L1_c16_run.lst --events --skip-score
run_pair rg_L1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 8200 --meta $LONG/L1_c16_run.lst --events --skip-score
run_pair rg_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 10000 --meta $LONG/L3_c16_run.lst --events --skip-score
run_pair rg_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 11200 --meta $LONG/L3_c16_run.lst --events --skip-score
for label in rg_L1_c16_a rg_L1_c16_b rg_L3_c16_a rg_L3_c16_b; do
  python3 $CHAIN $PAIR_ROOT/$label/a/events $PAIR_ROOT/$label/b/events > $D/chain_$label.txt 2>&1
  echo "$label preprocessing median a / b: $(grep 'request_admission -> tts_engine.stage_input_received' $D/chain_$label.txt | cut -d'|' -f4 | tr '\n' '/')" >> $L
done
LABELS="rg_s_c1 rg_L1_c16_a rg_L1_c16_b rg_L3_c16_a rg_L3_c16_b"
python3 $K/v3_20261020/pair_table.py $LABELS > $D/table_refgraphs.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py $LABELS > $D/paired_refgraphs.txt 2>&1
echo "queue_refgraphs_done $(date +%T)" >> $L
