#!/usr/bin/env bash
# queue_prs.sh after the review fixes (D4 the gdc import scoped to CUDA, D8 the device mel scoped to CUDA with the
# clip's copy to the device outside the lock, D2 comments): waits for queue_prs.sh's pair in flight (D4's seeded c1
# on the pre-fix D4 tree, the same CUDA code), then units on the fixed trees, the D8 speaker bench on the fixed tree
# with its embeddings compared to the pre-fix tree's and main's, and the pairs queue_prs.sh had left, on the fixed
# trees: D4 c16 and c64 in both orders and non-streaming seeded c1 and c16; D8 seeded c1, c16 and c64 in both orders,
# L1 c16 and L3 c16 in both orders, non-streaming seeded c1 and c16; D2 seeded c1 at seeds 1234 and 4321, c16 and
# c64 in both orders.
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
P4=${P4:?the fixed D4 tree}
P8=${P8:?the fixed D8 tree}
P2=${P2:?the fixed D2 tree}
O=tests/unit_test/qwen3_omni
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
while pgrep -f "v3_20261020/pair.py" > /dev/null; do sleep 10; done
echo "p4s_c1 rc - $(date +%T) (pre-fix D4 tree) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/p4s_c1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/p4s_c1/b/serve.log) $(grep 'byte identical' $D/p4s_c1.out)" >> $L
echo "queue prs2 start $(date +%T) d4 $(git -C $P4 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d8 $(git -C $P8 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d2 $(git -C $P2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
units() {
  local name=$1 tree=$2
  shift 2
  (cd $tree && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$tree timeout 3600 python3 -m pytest "$@" -q -p no:cacheprovider -rfEs \
    > $D/units/$name.log 2>&1)
  local rc=$?
  echo "units $name rc $rc $(grep -E 'passed|failed' $D/units/$name.log | tail -n 1) $(date +%T)" >> $L
  return $rc
}
units d4_fixed $P4 tests/unit_test/qwen3_tts
OK4=$?
units d8_fixed $P8 tests/unit_test/qwen3_tts
OK8=$?
units d2_fixed $P2 tests/unit_test/qwen3_tts tests/unit_test/kernels $O/test_code2wav.py $O/test_code2wav_cuda_graph.py \
  $O/test_code2wav_batching.py $O/test_code2wav_overlap.py $O/test_code2wav_snake_beta.py
OK2=$?
if [ $OK8 -eq 0 ]; then
  echo "== speaker bench seedtts d8 fixed" >> $D/speaker_bench.log
  PYTHONPATH=$P8 python3 $HERE/speaker_embed_bench.py $D/speaker_seedtts_d8fixed.pt --card 3 2>&1 \
    | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log
  echo "== speaker bench L3 d8 fixed" >> $D/speaker_bench.log
  PYTHONPATH=$P8 python3 $HERE/speaker_embed_bench.py $D/speaker_L3_d8fixed.pt --card 3 --repeats 5 --meta $LONG/L3_c16_run.lst 2>&1 \
    | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log
  for set in seedtts L3; do
    echo "== $set d8 / d8 fixed" >> $D/speaker_bench.log
    PYTHONPATH=$P8 python3 $HERE/speaker_embed_bench.py --compare $D/speaker_${set}_d8.pt $D/speaker_${set}_d8fixed.pt 2>&1 \
      | grep -v torchada >> $D/speaker_bench.log
    echo "== $set main / d8 fixed" >> $D/speaker_bench.log
    PYTHONPATH=$P8 python3 $HERE/speaker_embed_bench.py --compare $D/speaker_${set}_main.pt $D/speaker_${set}_d8fixed.pt 2>&1 \
      | grep -v torchada >> $D/speaker_bench.log
  done
  echo "bench d8 fixed done $(date +%T)" >> $L
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
echo "queue_prs2_done $(date +%T)" >> $L
