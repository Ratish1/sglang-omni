#!/usr/bin/env bash
# D1 (a long reference's first vocoder decode from its 580-frame reach) cut from main 3a4e73436, against main on the
# paths it touches only, cards 2 and 3: qwen3_tts units and the real-checkpoint gates (a failure runs nothing else),
# first_chunk_bench.py main, D1, D1, main; seeded L1 c1 (codes held equal), L1 c16 and L3 c16 in both card orders with
# request events and each pair's first-audio chain. SeedTTS EN is not run: its references are at most 111 frames, under
# the reach, so every first plan starts at frame 0 as on main.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
CHAIN=$T/tasks/qwen3-tts-perf/box/profile_20261022/first_audio_chain.py
LONG=/data/prof_20261022/formal/meta_w2/long
M=${M:?the main tree}
P1=${P1:?the D1 tree}
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
BASES="10000 11200 12400 13600 17000"
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
echo "queue d1 start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d1 $(git -C $P1 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_d1_done $(date +%T)" >> $L
  exit 1
else
  :
fi
mkdir -p $D/units
(cd $P1 && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$P1 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfEs \
  > $D/units/d1.log 2>&1)
UNITS=$?
echo "units d1 rc $UNITS $(grep -E 'passed|failed' $D/units/d1.log | tail -n 1) $(date +%T)" >> $L
(cd $P1 && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$P1 timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
  tests/unit_test/qwen3_tts/test_pipeline.py -m benchmark -k "test_real_" -q -p no:cacheprovider -rfEs \
  > $D/units/gates_d1.log 2>&1)
GATES=$?
echo "gates d1 rc $GATES $(grep -E 'passed|failed' $D/units/gates_d1.log | tail -n 1) $(date +%T)" >> $L
if [ $UNITS -ne 0 ] || [ $GATES -ne 0 ]; then
  echo "stopped: d1 units or gates failed" >> $L
  echo "queue_d1_done $(date +%T)" >> $L
  exit 1
else
  :
fi
for arm in "main $M" "d1 $P1" "d1 $P1" "main $M"; do
  set -- $arm
  echo "== first chunk bench $1 $(date +%T)" >> $D/first_chunk_bench_d1.log
  PYTHONPATH=$2 python3 $HERE/first_chunk_bench.py --card 3 >> $D/first_chunk_bench_d1.log 2>&1
done
echo "first chunk bench done $(date +%T)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $IN/pair.py --label $label --tree $M --b-tree $P1 "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair d1s_L1_c1 --concurrency 1 --a-card 2 --b-card 3 --port 10000 --seed 1234 --meta $LONG/L1_c1_run.lst
python3 $SEEDED/seeded_quality.py d1s_L1_c1 > $D/seeded_quality_d1.txt 2>&1
run_pair d1_L1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 11200 --meta $LONG/L1_c16_run.lst --events
run_pair d1_L1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 12400 --meta $LONG/L1_c16_run.lst --events
run_pair d1_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 13600 --meta $LONG/L3_c16_run.lst --events
run_pair d1_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 17000 --meta $LONG/L3_c16_run.lst --events
for label in d1_L1_c16_a d1_L1_c16_b d1_L3_c16_a d1_L3_c16_b; do
  python3 $CHAIN $PAIR_ROOT/$label/a/events $PAIR_ROOT/$label/b/events > $D/chain_$label.txt 2>&1
done
LABELS="d1s_L1_c1 d1_L1_c16_a d1_L1_c16_b d1_L3_c16_a d1_L3_c16_b"
python3 $IN/pair_table.py $LABELS > $D/table_d1.txt 2>&1
python3 $READ/paired_by_prompt.py $LABELS > $D/paired_d1.txt 2>&1
echo "queue_d1_done $(date +%T)" >> $L
