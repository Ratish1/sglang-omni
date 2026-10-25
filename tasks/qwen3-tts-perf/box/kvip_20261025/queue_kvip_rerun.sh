#!/usr/bin/env bash
# After queue_kvip_final.sh: kvf_c16_b's rings arm capped one request (a runaway, unseeded sampling) and from about
# 16:00 another container's CosyVoice benchmark loads cards 0 and 1 (about 350 W each), so c16 runs again in both
# card orders on the PR head's code, then units and the real-checkpoint tests on the PR head's exact tree (9493130e6:
# e6cbe020a plus two comments, a docstring and the slot count read from scratch_slot).
set -u
D=/data/kvip1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
M=${M:?the main tree}
P=${P:?the measured final tree}
H=${H:?the PR head tree}
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
until grep -q queue_kvip_final_done $L; do
  sleep 20
done
echo "queue kvip rerun start $(date +%T) main $(git -C $M rev-parse HEAD) final $(git -C $P rev-parse HEAD) head $(git -C $H rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $K/v3_20261020/pair.py --label $label --tree $M --b-tree $P "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
}
run_pair kvf2_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 7000
run_pair kvf2_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 8200
python3 $K/v3_20261020/pair_table.py kvf2_c16_a kvf2_c16_b > $D/table_kvf2.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py kvf2_c16_a kvf2_c16_b > $D/paired_kvf2.txt 2>&1
(cd $H && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$H timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q \
  -p no:cacheprovider -rfEs > $D/units/head.log 2>&1)
echo "units head rc $? $(grep -E 'passed|failed' $D/units/head.log | tail -n 1) $(date +%T)" >> $L
(cd $H && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$H timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark -k "test_real_" -q \
  -p no:cacheprovider -rfEs > $D/units/gates_head.log 2>&1)
echo "gates head rc $? $(grep -E 'passed|failed' $D/units/gates_head.log | tail -n 1) $(date +%T)" >> $L
echo "queue_kvip_rerun_done $(date +%T)" >> $L
