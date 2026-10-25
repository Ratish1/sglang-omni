#!/usr/bin/env bash
# The vocoder transformer's K/V in per-slot rings (perf/qwen3-tts-vocoder-kv-in-place) against its base, upstream main
# ab8193645, cards 2 and 3 (kv_in_place_20261025/DESIGN.md 7). Gate first: the ring unit test with the fused tests,
# the qwen3_tts units, the real-checkpoint tests; a failure stops the queue. Then the decode bench (main, rings,
# rings, main, card 3), the served-path distance to the fp32 decode on L1's references (compiled as served, main card
# 2 and rings card 3 at once), the standard SeedTTS EN set (seeded c1 both modes, c16 and c64 both card orders,
# non-streaming c16, unseeded pairs scored) and L3 c16 both orders (tail widths, long contexts).
set -u
D=/data/kvip1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
M=${M:?the main tree}
P=${P:?the rings tree}
LONG=/data/next1025/long
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
mkdir -p $D/units
BASES="4400 7000 8200 10000 11200 12400 13600 14800 17000"
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
echo "queue kvip start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')rings $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY]" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_kvip_done $(date +%T)" >> $L
  exit 1
else
  :
fi
stop() {
  echo "stopped: $1 $(date +%T)" >> $L
  echo "queue_kvip_done $(date +%T)" >> $L
  exit 1
}
(cd $P && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$P timeout 1200 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
  -k "ring_steps or fused_transformer" -q -p no:cacheprovider -rfEs > $D/units/ring.log 2>&1)
rc=$?
echo "ring test rc $rc $(grep -E 'passed|failed' $D/units/ring.log | tail -n 1) $(date +%T)" >> $L
[ $rc -eq 0 ] || stop "ring test"
(cd $P && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$P timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q \
  -p no:cacheprovider -rfEs > $D/units/rings.log 2>&1)
rc=$?
echo "units rings rc $rc $(grep -E 'passed|failed' $D/units/rings.log | tail -n 1) $(date +%T)" >> $L
[ $rc -eq 0 ] || stop "units"
(cd $P && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$P timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark -k "test_real_" -q \
  -p no:cacheprovider -rfEs > $D/units/gates_rings.log 2>&1)
rc=$?
echo "gates rings rc $rc $(grep -E 'passed|failed' $D/units/gates_rings.log | tail -n 1) $(date +%T)" >> $L
[ $rc -eq 0 ] || stop "gates"
for arm in "main $M" "rings $P" "ringsb $P" "mainb $M"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 timeout 1800 python3 $HERE/vocoder_decode_bench.py --card 3 > $D/bench_$1.txt 2>&1)
  echo "bench $1 rc $? $(head -n 1 $D/bench_$1.txt) $(date +%T)" >> $L
done
(cd $M && PYTHONPATH=$M timeout 3600 python3 $HERE/vocoder_truth_probe.py run $LONG/L1_c16_run.lst $D/truth_main.json \
  --card 2 --compiled > $D/truth_main.txt 2>&1) &
(cd $P && PYTHONPATH=$P timeout 3600 python3 $HERE/vocoder_truth_probe.py run $LONG/L1_c16_run.lst $D/truth_rings.json \
  --card 3 --compiled > $D/truth_rings.txt 2>&1) &
wait
(cd $M && PYTHONPATH=$M python3 $HERE/vocoder_truth_probe.py compare $D/truth_main.json $D/truth_rings.json \
  > $D/truth_compare.txt 2>&1)
echo "truth compare rc $? $(grep -m1 'rings' $D/truth_rings.txt) $(date +%T)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $K/v3_20261020/pair.py --label $label --tree $M --b-tree $P "$@" > $D/$label.out 2>&1
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
      python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 > $D/similarity_$label.out 2>&1
      echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
    fi
  done
}
run_pair kv_s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 4400 --seed 1234
run_pair kv_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 7000
run_pair kv_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 8200
run_pair kv_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 10000
run_pair kv_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 11200
run_pair kv_ns_c1 --concurrency 1 --a-card 3 --b-card 2 --port 12400 --seed 1234 --no-stream
run_pair kv_ns_c16 --concurrency 16 --a-card 2 --b-card 3 --port 13600 --no-stream
run_pair kv_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 14800 --meta $LONG/L3_c16_run.lst --events --skip-score
run_pair kv_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 17000 --meta $LONG/L3_c16_run.lst --events --skip-score
similarity kv_s_c1 kv_c16_a kv_c16_b kv_c64_a kv_c64_b kv_ns_c1 kv_ns_c16
LABELS="kv_s_c1 kv_c16_a kv_c16_b kv_c64_a kv_c64_b kv_ns_c1 kv_ns_c16 kv_L3_c16_a kv_L3_c16_b"
python3 $K/v3_20261020/pair_table.py $LABELS > $D/table_kv.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py kv_s_c1 kv_c16_a kv_c16_b kv_c64_a kv_c64_b kv_L3_c16_a kv_L3_c16_b \
  > $D/paired_kv.txt 2>&1
python3 $K/q3t_20261022/first_audio_tail.py kv_c16_a kv_c16_b kv_c64_a kv_c64_b > $D/first_audio_tail_kv.txt 2>&1
(cd $M && PYTHONPATH=$M python3 $K/d1merge_20261024/ci_metrics.py kv_s_c1 kv_c16_a kv_c16_b kv_c64_a kv_c64_b kv_ns_c1 \
  kv_ns_c16 > $D/ci_metrics_kv.txt 2>&1)
echo "queue_kvip_done $(date +%T)" >> $L
