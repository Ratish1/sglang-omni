#!/usr/bin/env bash
# The three PR branches cut from main after #2999's merge, each against main alone: D4 (the predictor's codebook
# tail), D8 (the reference preprocessing on the device and at recurring widths), D2 with D9 (the vocoder's slot conv
# stack). Units first on every branch, then each branch's component bench and SeedTTS EN pairs, streaming unless
# named: D4 seeded c1 (byte identity) and unseeded c16 and c64 in both card orders, non-streaming seeded c1 and c16;
# D8 seeded c1, c16 and c64 in both orders, the long cells L1 c16 and L3 c16 in both orders, non-streaming seeded c1
# and c16; D2 seeded c1 at seeds 1234 and 4321 (codes held equal) and c16 and c64 in both orders. A branch whose units
# fail runs nothing else.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
M=${M:?the main tree}
P4=${P4:?the D4 tree}
P8=${P8:?the D8 tree}
P2=${P2:?the D2 tree}
O=tests/unit_test/qwen3_omni
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
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
echo "queue prs start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d4 $(git -C $P4 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d8 $(git -C $P8 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d2 $(git -C $P2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY] loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
if [ -n "$BUSY" ]; then
  echo "stopped: listeners inside the pair port ranges" >> $L
  echo "queue_prs_done $(date +%T)" >> $L
  exit 1
fi
units() {
  local name=$1 tree=$2
  shift 2
  (cd $tree && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$tree timeout 3600 python3 -m pytest "$@" -q -p no:cacheprovider -rfEs \
    > $D/units/$name.log 2>&1)
  local rc=$?
  echo "units $name rc $rc $(grep -E 'passed|failed' $D/units/$name.log | tail -n 1) $(date +%T)" >> $L
  return $rc
}
units d4 $P4 tests/unit_test/qwen3_tts
OK4=$?
units d8 $P8 tests/unit_test/qwen3_tts
OK8=$?
units d2 $P2 tests/unit_test/qwen3_tts tests/unit_test/kernels $O/test_code2wav.py $O/test_code2wav_cuda_graph.py \
  $O/test_code2wav_batching.py $O/test_code2wav_overlap.py $O/test_code2wav_snake_beta.py
OK2=$?
(cd $P2 && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$P2 timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
  tests/unit_test/qwen3_tts/test_pipeline.py -m benchmark -k "test_real_" -q -p no:cacheprovider -rfEs \
  > $D/units/gates_d2.log 2>&1)
GATES2=$?
echo "gates d2 rc $GATES2 $(grep -E 'passed|failed' $D/units/gates_d2.log | tail -n 1) $(date +%T)" >> $L
if [ $GATES2 -ne 0 ]; then
  OK2=1
else
  :
fi
if [ $OK8 -eq 0 ]; then
  for arm in "main $M" "d8 $P8"; do
    set -- $arm
    echo "== speaker bench seedtts $1" >> $D/speaker_bench.log
    PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_seedtts_$1.pt --card 3 2>&1 \
      | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log
    echo "== speaker bench L3 $1" >> $D/speaker_bench.log
    PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_L3_$1.pt --card 3 --repeats 5 --meta $LONG/L3_c16_run.lst 2>&1 \
      | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log
    echo "== reference batcher $1" >> $D/reference_batcher_bench.log
    PYTHONPATH=$2 python3 $HERE/reference_batcher_bench.py --card 3 --warm $LONG/L3_c16_warm.lst --run $LONG/L3_c16_run.lst \
      >> $D/reference_batcher_bench.log 2>&1
  done
  echo "== seedtts main / d8" >> $D/speaker_bench.log
  PYTHONPATH=$P8 python3 $HERE/speaker_embed_bench.py --compare $D/speaker_seedtts_main.pt $D/speaker_seedtts_d8.pt >> $D/speaker_bench.log 2>&1
  echo "== L3 main / d8" >> $D/speaker_bench.log
  PYTHONPATH=$P8 python3 $HERE/speaker_embed_bench.py --compare $D/speaker_L3_main.pt $D/speaker_L3_d8.pt >> $D/speaker_bench.log 2>&1
  echo "bench d8 done $(date +%T)" >> $L
else
  :
fi
if [ $OK2 -eq 0 ]; then
  for arm in "main $M" "d2 $P2" "d2 $P2" "main $M"; do
    set -- $arm
    echo "== bench $1" >> $D/bench_d2.log
    PYTHONPATH=$2 python3 $HERE/vocoder_decode_bench.py --card 3 2>&1 | grep -E "^(cold|warm|tail) " >> $D/bench_d2.log
  done
  echo "bench d2 done $(date +%T)" >> $L
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
  local prefix=$1
  local labels=$(ls $PAIR_ROOT | grep "^$prefix" | sort)
  python3 $IN/pair_table.py $labels > $D/table_$prefix.txt 2>&1
  python3 $READ/paired_by_prompt.py $labels > $D/paired_$prefix.txt 2>&1
  python3 $READ/first_audio_tail.py $labels > $D/first_audio_tail_$prefix.txt 2>&1
}
if [ $OK4 -eq 0 ]; then
  run_pair p4s_c1 $P4 --concurrency 1 --a-card 2 --b-card 3 --port 10000 --seed 1234
  run_pair p4_c16_a $P4 --concurrency 16 --a-card 2 --b-card 3 --port 11200
  run_pair p4_c16_b $P4 --concurrency 16 --a-card 3 --b-card 2 --port 12400
  run_pair p4_c64_a $P4 --concurrency 64 --a-card 2 --b-card 3 --port 13600
  run_pair p4_c64_b $P4 --concurrency 64 --a-card 3 --b-card 2 --port 17000
  run_pair p4ns_c1 $P4 --concurrency 1 --a-card 3 --b-card 2 --port 18200 --seed 1234 --no-stream
  run_pair p4ns_c16 $P4 --concurrency 16 --a-card 2 --b-card 3 --port 19400 --no-stream
  tables p4
  echo "d4 done $(date +%T)" >> $L
else
  :
fi
if [ $OK8 -eq 0 ]; then
  run_pair p8s_c1 $P8 --concurrency 1 --a-card 2 --b-card 3 --port 20600 --seed 1234
  run_pair p8_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 21800
  run_pair p8_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 23000
  run_pair p8_c64_a $P8 --concurrency 64 --a-card 2 --b-card 3 --port 24200
  run_pair p8_c64_b $P8 --concurrency 64 --a-card 3 --b-card 2 --port 25400
  run_pair p8_L1_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 26600 --meta $LONG/L1_c16_run.lst
  run_pair p8_L1_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 27800 --meta $LONG/L1_c16_run.lst
  run_pair p8_L3_c16_a $P8 --concurrency 16 --a-card 2 --b-card 3 --port 10000 --meta $LONG/L3_c16_run.lst
  run_pair p8_L3_c16_b $P8 --concurrency 16 --a-card 3 --b-card 2 --port 11200 --meta $LONG/L3_c16_run.lst
  run_pair p8ns_c1 $P8 --concurrency 1 --a-card 3 --b-card 2 --port 12400 --seed 1234 --no-stream
  run_pair p8ns_c16 $P8 --concurrency 16 --a-card 2 --b-card 3 --port 13600 --no-stream
  tables p8
  echo "d8 done $(date +%T)" >> $L
else
  :
fi
if [ $OK2 -eq 0 ]; then
  run_pair p2s_c1_1234 $P2 --concurrency 1 --a-card 2 --b-card 3 --port 17000 --seed 1234
  run_pair p2s_c1_4321 $P2 --concurrency 1 --a-card 3 --b-card 2 --port 18200 --seed 4321
  python3 $SEEDED/seeded_quality.py p2s_c1_1234 p2s_c1_4321 > $D/seeded_quality_p2.txt 2>&1
  run_pair p2_c16_a $P2 --concurrency 16 --a-card 2 --b-card 3 --port 19400
  run_pair p2_c16_b $P2 --concurrency 16 --a-card 3 --b-card 2 --port 20600
  run_pair p2_c64_a $P2 --concurrency 64 --a-card 2 --b-card 3 --port 21800
  run_pair p2_c64_b $P2 --concurrency 64 --a-card 3 --b-card 2 --port 23000
  tables p2
  echo "d2 done $(date +%T)" >> $L
else
  :
fi
echo "queue_prs_done $(date +%T)" >> $L
