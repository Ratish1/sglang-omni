#!/usr/bin/env bash
# Speed and gates after queue_omni3097.sh: the code2wav bias SnakeBeta launches of one decode, head and new in ABBA on
# card 2; Qwen3-Omni seeded c16 and c64 streamed in both card orders (codes held equal, so both arms do the same work),
# unscored; then the trained Qwen3-TTS gate and Qwen3-Omni's real code2wav gate on the rebuilt head.
set -u
D=/data/omni3097
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
OLD=${OLD:?the head tree}
NEW=${NEW:?the new tree}
GATE=${GATE:?the rebuilt head tree}
HERE=$T/tasks/qwen3-tts-perf/box/review3097_20261023
L=$D/queue.log
echo "queue omni3097b start $(date +%T) kit $(git -C $T rev-parse HEAD) gate $(git -C $GATE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
for arm in old new new old old new new old; do
  if [ $arm = old ]; then tree=$OLD; else tree=$NEW; fi
  echo "snake bench $arm $(cd $D && PYTHONPATH=$tree python3 $HERE/snake_kernel_bench.py --card 2 2>&1 | tail -n 1) $(date +%T)" >> $L
done
python3 $HERE/omni_pair.py --label o3 --a-tree $OLD --b-tree $NEW --a-card 2 --b-card 3 --port 20000 --passes s_c16_seeded,s_c64_seeded --score none > $D/o3.out 2>&1
echo "omni o3 rc $? $(date +%T)" >> $L
python3 $HERE/omni_pair.py --label o4 --a-tree $OLD --b-tree $NEW --a-card 3 --b-card 2 --port 22000 --passes s_c16_seeded,s_c64_seeded --score none > $D/o4.out 2>&1
echo "omni o4 rc $? $(date +%T)" >> $L
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
(cd $GATE && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer python3 -m pytest -q --disable-warnings --tb=short -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec.py -m benchmark -k test_real_) > $D/gate_rebuilt.out 2>&1
echo "tts gate rebuilt rc $? $(tail -n 1 $D/gate_rebuilt.out) $(date +%T)" >> $L
for tree in $OLD $GATE; do
  (cd $tree && CUDA_VISIBLE_DEVICES=2,3 QWEN3_OMNI_MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct python3 -m pytest -q --disable-warnings --tb=short -p no:cacheprovider tests/unit_test/qwen3_omni/test_code2wav_snake_beta.py -k test_real_) > $D/omni_gate_$(basename $tree).out 2>&1
  echo "omni gate $(basename $tree) rc $? $(tail -n 1 $D/omni_gate_$(basename $tree).out) $(date +%T)" >> $L
done
echo "queue_omni3097b_done $(date +%T)" >> $L
