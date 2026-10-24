#!/usr/bin/env bash
# D8 version 2 (7e06fc2c4): the speaker mel stays on the CPU inside the graphs (version 1's in-graph mel added 58 us
# and 21 kernels of device time to every SeedTTS embed for no first-audio gain, and read +0.2 to +0.4 % c16 RTF beyond
# the card offset); past the graphs the device mel and the rounded widths are version 1's code. Units and gates with
# the 0.6B path; the speaker bench and its embeddings against main (SeedTTS EN expected identical); the speaker device
# time; SeedTTS EN seeded c1 and non-streaming seeded c1 (byte identity expected on all 1,088), c16 and c64 in both card
# orders and non-streaming c16, unscored; L1 c16 and L3 c16 in both orders with request events, L1 scored and its SIM
# against each sample's own clip. Cards 2 and 3.
set -u
D=/data/next1024
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
CHAIN=$K/profile_20261022/first_audio_chain.py
LONG=/data/d1merge/long
SHORT=/data/d1merge/short_reference_meta
M=${M:?the main tree}
P8=${P8:?the D8 version 2 tree}
SNAP17=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
SNAP06=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/* | head -n 1)
L=$D/queue.log
echo "queue d8v2 start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d8v2 $(git -C $P8 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $P8 && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$P8 timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider \
  -rfEs > $D/units/d8v2.log 2>&1)
echo "units d8v2 rc $? $(grep -E 'passed|failed' $D/units/d8v2.log | tail -n 1) $(date +%T)" >> $L
(cd $P8 && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP17/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP17 \
  QWEN3_TTS_BASE_0_6B_PATH=$SNAP06 PYTHONPATH=$P8 timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark \
  -k "test_real_" -q -p no:cacheprovider -rfEs > $D/units/gates_all_d8v2.log 2>&1)
echo "gates d8v2 rc $? $(grep -E 'passed|failed' $D/units/gates_all_d8v2.log | tail -n 1) $(date +%T)" >> $L
for arm in "d8v2 $P8" "d8v2b $P8"; do
  set -- $arm
  echo "== speaker bench seedtts $1" >> $D/speaker_bench.log
  (cd $2 && PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_seedtts_$1.pt --card 3 2>&1 \
    | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log)
  echo "== speaker bench L3 $1" >> $D/speaker_bench.log
  (cd $2 && PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_L3_$1.pt --card 3 --repeats 5 \
    --meta $LONG/L3_c16_run.lst 2>&1 | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log)
done
for set_name in seedtts L3; do
  for pair in "main d8v2" "d8 d8v2"; do
    set -- $pair
    (cd $M && PYTHONPATH=$M python3 $HERE/speaker_embed_bench.py --compare $D/speaker_${set_name}_$1.pt \
      $D/speaker_${set_name}_$2.pt > $D/speaker_compare_${set_name}_$1_$2.txt 2>&1)
    echo "speaker $set_name $1 against $2: $(grep cosine $D/speaker_compare_${set_name}_$1_$2.txt)" >> $L
  done
done
(cd $P8 && PYTHONPATH=$P8 python3 $K/next_20261024/speaker_device_time.py --card 3 > $D/speaker_time_d8v2.txt 2>&1)
echo "speaker device time d8v2: $(grep 'speaker embed' $D/speaker_time_d8v2.txt) $(date +%T)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $K/v3_20261020/pair.py --label $label --tree $M --b-tree $P8 "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair d8v2s_c1 --concurrency 1 --a-card 2 --b-card 3 --port 4400 --seed 1234 --skip-score
run_pair d8v2_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 7000 --skip-score
run_pair d8v2_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 8200 --skip-score
run_pair d8v2_c64_a --concurrency 64 --a-card 2 --b-card 3 --port 10000 --skip-score
run_pair d8v2_c64_b --concurrency 64 --a-card 3 --b-card 2 --port 11200 --skip-score
run_pair d8v2ns_c1 --concurrency 1 --a-card 3 --b-card 2 --port 12400 --seed 1234 --no-stream --skip-score
run_pair d8v2ns_c16 --concurrency 16 --a-card 2 --b-card 3 --port 13600 --no-stream --skip-score
run_pair d8v2_L1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 14800 --meta $LONG/L1_c16_run.lst --events
run_pair d8v2_L1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 17000 --meta $LONG/L1_c16_run.lst --events
run_pair d8v2_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 18200 --meta $LONG/L3_c16_run.lst --events --skip-score
run_pair d8v2_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 19400 --meta $LONG/L3_c16_run.lst --events --skip-score
for label in d8v2_L1_c16_a d8v2_L1_c16_b d8v2_L3_c16_a d8v2_L3_c16_b; do
  python3 $CHAIN $PAIR_ROOT/$label/a/events $PAIR_ROOT/$label/b/events > $D/chain_$label.txt 2>&1
done
for label in d8v2_L1_c16_a d8v2_L1_c16_b; do
  python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 --meta $SHORT/L1_c16_run.lst > $D/similarity_$label.out 2>&1
  echo "similarity excluding caps $label rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
done
LABELS="d8v2s_c1 d8v2_c16_a d8v2_c16_b d8v2_c64_a d8v2_c64_b d8v2ns_c1 d8v2ns_c16 d8v2_L1_c16_a d8v2_L1_c16_b d8v2_L3_c16_a d8v2_L3_c16_b"
python3 $K/v3_20261020/pair_table.py $LABELS > $D/table_d8v2.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py d8v2s_c1 d8v2_c16_a d8v2_c16_b d8v2_c64_a d8v2_c64_b d8v2_L1_c16_a d8v2_L1_c16_b \
  d8v2_L3_c16_a d8v2_L3_c16_b > $D/paired_d8v2.txt 2>&1
python3 $K/q3t_20261022/first_audio_tail.py d8v2_c16_a d8v2_c16_b d8v2_c64_a d8v2_c64_b > $D/first_audio_tail_d8v2.txt 2>&1
echo "queue_d8v2_done $(date +%T)" >> $L
