#!/usr/bin/env bash
# After queue_kvip.sh: the final head (9180d5ec6: the rings sized to the widest cold decode too, plus the arena ring
# test) gated again (units, real-checkpoint tests with the new arena test), its decode bench against main (main,
# rings, rings, main, card 3, the ring sized as served for the bench's cold widths 8 and 9), and the cause of the
# compiled wide decode's batch-1 cost on the measured head (cold width 9 through the gathered step under compile with
# an 8-frame ring), read from per-kernel tables of both arms. Cards 2 and 3, after the pairs, so no timing overlaps.
set -u
D=/data/kvip1025
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=${M:?the main tree}
F=${F:?the final rings tree}
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
until grep -q queue_kvip_done $L; do
  sleep 30
done
echo "queue kvip after start $(date +%T) final $(git -C $F rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $F && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$F timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q \
  -p no:cacheprovider -rfEs > $D/units/final.log 2>&1)
echo "units final rc $? $(grep -E 'passed|failed' $D/units/final.log | tail -n 1) $(date +%T)" >> $L
(cd $F && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$F timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark -k "test_real_" -q \
  -p no:cacheprovider -rfEs > $D/units/gates_final.log 2>&1)
echo "gates final rc $? $(grep -E 'passed|failed' $D/units/gates_final.log | tail -n 1) $(date +%T)" >> $L
for arm in "main2 $M" "final $F" "finalb $F" "main2b $M"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 timeout 1800 python3 $HERE/vocoder_decode_bench.py --card 3 > $D/bench_$1.txt 2>&1)
  echo "bench $1 rc $? $(grep -m1 'rings:' $D/bench_$1.txt) $(date +%T)" >> $L
done
(cd $M && PYTHONPATH=$M timeout 1800 python3 $HERE/vocoder_decode_bench.py --card 3 --kernel-table \
  > $D/bench_table_main.txt 2>&1)
echo "kernel table main rc $? $(date +%T)" >> $L
(cd $F && PYTHONPATH=$F timeout 1800 python3 $HERE/vocoder_decode_bench.py --card 3 --kernel-table --ring-frames 8 \
  > $D/bench_table_ring8.txt 2>&1)
echo "kernel table ring 8 rc $? $(date +%T)" >> $L
echo "queue_kvip_after_done $(date +%T)" >> $L
