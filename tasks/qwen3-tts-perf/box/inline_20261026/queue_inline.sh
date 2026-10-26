#!/usr/bin/env bash
# The inline commit b4b7331c2 (the speaker mel's padding, frame count and zero tail written inline) against #3177's head
# 52983c59d, then the long-reference memory edge, main against the PR; one H200, card 0, one arm at a time.
# Inline is confirmed when its units and the real-checkpoint gates pass, its speaker embeddings equal the PR's with max
# |delta| 0 on SeedTTS EN (64), the L3 references (about 88 s) and the 200 s references, and seeded SeedTTS EN c1 is
# byte identical to the PR's on all 1,088 in both serving modes. The memory read: 200 s references at c8, main, PR, main
# (the card's peak and its rise over health, failed requests, allocation failures in the serve log), then the L1 88 s
# references at c16, main and PR; the PR's peak beyond main's two runs' spread opens a chunked device STFT.
set -u
D=/data/inline_20261026
export ARM_ROOT=$D/arms
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/inline_20261026
BENCH=$K/chain_20261022/speaker_embed_bench.py
M=${M:?the main tree}
P=${P:?the PR tree}
I=${I:?the inline tree}
L=$D/queue.log
LONG=$D/long
REF200=$D/ref200
mkdir -p $D/units $LONG $REF200
PORTS="4400 4600 4800 5000 7000 7200 7400 7600 7800"
BUSY=$(python3 - $PORTS <<'EOF'
import sys
listening = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(table).read().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            listening.add(int(fields[1].rsplit(":", 1)[1], 16))
print(" ".join(str(port) for port in sorted(listening) if any(int(base) <= port <= int(base) + 100 for base in sys.argv[1:])))
EOF
)
echo "queue inline start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')pr $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')inline $(git -C $I rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners on the arm ports" >> $L
  echo "queue_inline_done $(date +%T)" >> $L
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
(cd $M && PYTHONPATH=$M python3 $K/profile_20261020/long_cells_meta.py $LONG >> $L 2>&1)
(cd $M && PYTHONPATH=$M python3 $HERE/long_ref_seconds_meta.py $REF200 200 64 >> $L 2>&1)
echo "lists $(wc -l < $LONG/L1_c16_run.lst) L1 rows, $(wc -l < $LONG/L3_c16_run.lst) L3 rows, $(wc -l < $REF200/ref200s.lst) 200 s rows, longest $(cut -f3 $REF200/ref200s_lengths.tsv | sort -n | tail -n 1) s, largest $(cut -f5 $REF200/ref200s_lengths.tsv | sort -n | tail -n 1) bytes, rate $(cut -f4 $REF200/ref200s_lengths.tsv | sort -u | tail -n 1) $(date +%T)" >> $L
(cd $I && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$I timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider \
  -rfEs > $D/units/inline.log 2>&1)
echo "units inline rc $? $(grep -E 'passed|failed' $D/units/inline.log | tail -n 1) $(date +%T)" >> $L
(cd $I && CUDA_VISIBLE_DEVICES=0 QWEN3_TTS_TOKENIZER_PATH=$SNAP17/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP17 \
  QWEN3_TTS_BASE_0_6B_PATH=$SNAP06 PYTHONPATH=$I timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark \
  -k "test_real_" -q -p no:cacheprovider -rfEs > $D/units/gates_inline.log 2>&1)
echo "gates inline rc $? $(grep -E 'passed|failed' $D/units/gates_inline.log | tail -n 1) $(date +%T)" >> $L
for arm in "pr $P" "inline $I" "main $M"; do
  set -- $arm
  if [ $1 != main ]; then
    (cd $2 && PYTHONPATH=$2 python3 $BENCH $D/speaker_seedtts_$1.pt --card 0 > $D/speaker_seedtts_$1.txt 2>&1)
    echo "speaker seedtts $1: $(grep -E 'embed wall|Error' $D/speaker_seedtts_$1.txt | tail -n 1) $(date +%T)" >> $L
  else
    :
  fi
  (cd $2 && PYTHONPATH=$2 python3 $BENCH $D/speaker_L3_$1.pt --card 0 --repeats 5 --meta $LONG/L3_c16_run.lst \
    > $D/speaker_L3_$1.txt 2>&1)
  echo "speaker L3 $1: $(grep -E 'embed wall|Error' $D/speaker_L3_$1.txt | tail -n 1) $(date +%T)" >> $L
  (cd $2 && PYTHONPATH=$2 python3 $BENCH $D/speaker_ref200_$1.pt --card 0 --repeats 5 --meta $REF200/ref200s.lst \
    > $D/speaker_ref200_$1.txt 2>&1)
  echo "speaker ref200 $1: $(grep -E 'embed wall|Error' $D/speaker_ref200_$1.txt | tail -n 1) $(date +%T)" >> $L
done
for compared in "seedtts pr inline" "L3 pr inline" "ref200 pr inline" "L3 main pr" "ref200 main pr"; do
  set -- $compared
  (cd $M && PYTHONPATH=$M python3 $BENCH --compare $D/speaker_$1_$2.pt $D/speaker_$1_$3.pt > $D/speaker_compare_$1_$2_$3.txt 2>&1)
  echo "speaker $1 $2 against $3: $(tail -n 1 $D/speaker_compare_$1_$2_$3.txt)" >> $L
done
run_arm() {
  python3 $HERE/serve_arm.py "$@" >> $L 2>&1
}
run_arm --label c1s_pr --tree $P --port 4400 --concurrency 1 --seed 1234
run_arm --label c1s_inline --tree $I --port 4600 --concurrency 1 --seed 1234
python3 $HERE/serve_arm.py --compare c1s_pr c1s_inline | sed 's/^/c1 streaming pr against inline: /' >> $L 2>&1
run_arm --label c1ns_inline --tree $I --port 4800 --concurrency 1 --seed 1234 --no-stream
run_arm --label c1ns_pr --tree $P --port 5000 --concurrency 1 --seed 1234 --no-stream
python3 $HERE/serve_arm.py --compare c1ns_pr c1ns_inline | sed 's/^/c1 non-streaming pr against inline: /' >> $L 2>&1
run_arm --label ref200_c8_main --tree $M --port 7000 --concurrency 8 --meta $REF200/ref200s.lst
run_arm --label ref200_c8_pr --tree $P --port 7200 --concurrency 8 --meta $REF200/ref200s.lst
run_arm --label ref200_c8_mainb --tree $M --port 7400 --concurrency 8 --meta $REF200/ref200s.lst
run_arm --label L1_c16_main --tree $M --port 7600 --concurrency 16 --meta $LONG/L1_c16_run.lst
run_arm --label L1_c16_pr --tree $P --port 7800 --concurrency 16 --meta $LONG/L1_c16_run.lst
echo "queue_inline_done $(date +%T)" >> $L
