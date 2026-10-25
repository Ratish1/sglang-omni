#!/usr/bin/env bash
# D8 version 3 (b17546a65, DECISIONS 95) on a fresh lease: the speaker mel inside the graphs is main's CPU mel (F.pad
# and the same spectral operations), past the graphs the device mel at rounded widths (version 1's code), and capture
# runs the device mel once at twice the largest bucket so cuFFT initializes at startup. First the lists and the two
# checkpoints; then units and the real-checkpoint gates with both checkpoints; the speaker bench main, D8, D8, main
# (SeedTTS EN wall within main's and embeddings bit identical; 88 s references as version 1, first calls near 11 ms);
# the speaker device time (main's 725 us expected); then, if the gates pass, SeedTTS EN seeded c1 and non-streaming
# seeded c1 (byte identity expected on all 1,088), c16 and c64 in both card orders and non-streaming c16, unscored;
# L1 c16 and L3 c16 in both orders with request events, L1 scored and its SIM against each sample's own clip. Cards 2
# and 3.
set -u
D=/data/next1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
PROFILE=$K/profile_20261020
CHAIN=$K/profile_20261022/first_audio_chain.py
LONG=$D/long
SHORT=$D/short_reference_meta
M=${M:?the main tree}
P8=${P8:?the D8 version 3 tree}
L=$D/queue.log
mkdir -p $D/units
BASES="4400 7000 8200 10000 11200 12400 13600 14800 17000 18200 19400"
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
echo "queue d8v3 start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d8v3 $(git -C $P8 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_d8v3_done $(date +%T)" >> $L
  exit 1
else
  :
fi
for model in Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-12Hz-0.6B-Base; do
  python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model'))" >> $D/download.log 2>&1
  echo "download $model rc $? $(date +%T)" >> $L
done
SNAP17=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
SNAP06=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/* | head -n 1)
(cd $M && PYTHONPATH=$M python3 $PROFILE/long_cells_meta.py $LONG >> $L 2>&1)
(cd $M && PYTHONPATH=$M python3 $PROFILE/short_reference_meta.py $LONG $SHORT L1_c1 L1_c16 >> $L 2>&1)
echo "lists $(wc -l < $LONG/L1_c16_run.lst) L1 rows, $(wc -l < $SHORT/L1_c16_run.lst) short rows $(date +%T)" >> $L
(cd $P8 && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$P8 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider \
  -rfEs > $D/units/d8v3.log 2>&1)
UNITS=$?
echo "units d8v3 rc $UNITS $(grep -E 'passed|failed' $D/units/d8v3.log | tail -n 1) $(date +%T)" >> $L
(cd $P8 && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP17/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP17 \
  QWEN3_TTS_BASE_0_6B_PATH=$SNAP06 PYTHONPATH=$P8 timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark \
  -k "test_real_" -q -p no:cacheprovider -rfEs > $D/units/gates_all_d8v3.log 2>&1)
GATES=$?
echo "gates d8v3 rc $GATES $(grep -E 'passed|failed' $D/units/gates_all_d8v3.log | tail -n 1) $(date +%T)" >> $L
for arm in "main $M" "d8v3 $P8" "d8v3b $P8" "mainb $M"; do
  set -- $arm
  echo "== speaker bench seedtts $1" >> $D/speaker_bench.log
  (cd $2 && PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_seedtts_$1.pt --card 3 2>&1 \
    | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log)
  echo "== speaker bench L3 $1" >> $D/speaker_bench.log
  (cd $2 && PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_L3_$1.pt --card 3 --repeats 5 \
    --meta $LONG/L3_c16_run.lst 2>&1 | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log)
  echo "speaker bench $1: $(grep -A1 "seedtts $1\$" $D/speaker_bench.log | grep -oE 'median [0-9.]+' | head -1) seedtts, $(grep -A1 "L3 $1\$" $D/speaker_bench.log | grep -oE 'median [0-9.]+|max [0-9.]+;' | tr '\n' ' ')L3 $(date +%T)" >> $L
done
for set_name in seedtts L3; do
  for pair in "main d8v3" "main mainb" "d8v3 d8v3b"; do
    set -- $pair
    (cd $M && PYTHONPATH=$M python3 $HERE/speaker_embed_bench.py --compare $D/speaker_${set_name}_$1.pt \
      $D/speaker_${set_name}_$2.pt > $D/speaker_compare_${set_name}_$1_$2.txt 2>&1)
    echo "speaker $set_name $1 against $2: $(grep cosine $D/speaker_compare_${set_name}_$1_$2.txt)" >> $L
  done
done
for arm in "main $M" "d8v3 $P8"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 python3 $K/next_20261024/speaker_device_time.py --card 3 > $D/speaker_time_$1.txt 2>&1)
  echo "speaker device time $1: $(grep 'speaker embed' $D/speaker_time_$1.txt) $(date +%T)" >> $L
done
if [ $UNITS -ne 0 ] || [ $GATES -ne 0 ]; then
  echo "pairs skipped: units or gates failed" >> $L
  echo "queue_d8v3_done $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  local label=$1
  shift
  python3 $K/v3_20261020/pair.py --label $label --tree $M --b-tree $P8 "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair d8v3s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 4400 --seed 1234 --skip-score
run_pair d8v3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 7000 --skip-score
run_pair d8v3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 8200 --skip-score
run_pair d8v3_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 10000 --skip-score
run_pair d8v3_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 11200 --skip-score
run_pair d8v3ns_c1 --concurrency 1 --a-card 3 --b-card 2 --port 12400 --seed 1234 --no-stream --skip-score
run_pair d8v3ns_c16 --concurrency 16 --a-card 2 --b-card 3 --port 13600 --no-stream --skip-score
run_pair d8v3_L1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 14800 --meta $LONG/L1_c16_run.lst --events
run_pair d8v3_L1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 17000 --meta $LONG/L1_c16_run.lst --events
run_pair d8v3_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 18200 --meta $LONG/L3_c16_run.lst --events --skip-score
run_pair d8v3_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 19400 --meta $LONG/L3_c16_run.lst --events --skip-score
for label in d8v3_L1_c16_a d8v3_L1_c16_b d8v3_L3_c16_a d8v3_L3_c16_b; do
  python3 $CHAIN $PAIR_ROOT/$label/a/events $PAIR_ROOT/$label/b/events > $D/chain_$label.txt 2>&1
done
for label in d8v3_L1_c16_a d8v3_L1_c16_b; do
  python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 --meta $SHORT/L1_c16_run.lst > $D/similarity_$label.out 2>&1
  echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
done
LABELS="d8v3s_c1 d8v3_c16_a d8v3_c16_b d8v3_c64_a d8v3_c64_b d8v3ns_c1 d8v3ns_c16 d8v3_L1_c16_a d8v3_L1_c16_b d8v3_L3_c16_a d8v3_L3_c16_b"
python3 $K/v3_20261020/pair_table.py $LABELS > $D/table_d8v3.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py d8v3s_c1 d8v3_c16_a d8v3_c16_b d8v3_c64_a d8v3_c64_b d8v3_L1_c16_a d8v3_L1_c16_b \
  d8v3_L3_c16_a d8v3_L3_c16_b > $D/paired_d8v3.txt 2>&1
python3 $K/q3t_20261022/first_audio_tail.py d8v3_c16_a d8v3_c16_b d8v3_c64_a d8v3_c64_b > $D/first_audio_tail_d8v3.txt 2>&1
echo "queue_d8v3_done $(date +%T)" >> $L
