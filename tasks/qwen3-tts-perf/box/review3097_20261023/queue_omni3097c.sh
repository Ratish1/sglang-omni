#!/usr/bin/env bash
# #3097 with code2wav's eager SnakeBeta rounding kept as the default (Qwen3-TTS passes fp32), against main for Qwen3-Omni
# and against the measured head for Qwen3-TTS: suites and both real gates; the TTS padded live-row hash; Omni's slot
# decode and its SnakeBeta launches main against the PR in ABBA on card 2; Omni seeded pairs main (card 2) against the
# PR (card 3), c1 streamed and not and c16 streamed, unscored, with every request's served audio compared byte for byte.
set -u
D=/data/omni3097
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
NEW=${NEW:?the PR tree}
HERE=$T/tasks/qwen3-tts-perf/box/review3097_20261023
L=$D/queue.log
echo "queue omni3097c start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')new $(git -C $NEW rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
SUITES="tests/unit_test/kernels/test_codec_state.py tests/unit_test/kernels/test_bias_snake_beta.py tests/unit_test/kernels/test_single_output_conv.py tests/unit_test/qwen3_tts/test_incremental_codec.py tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py tests/unit_test/qwen3_tts/test_pipeline.py tests/unit_test/qwen3_omni/test_code2wav.py tests/unit_test/qwen3_omni/test_code2wav_batching.py tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py tests/unit_test/qwen3_omni/test_code2wav_overlap.py tests/unit_test/qwen3_omni/test_code2wav_snake_beta.py tests/unit_test/qwen3_omni/test_codec_state_pool.py"
(cd $NEW && CUDA_VISIBLE_DEVICES=2,3 python3 -m pytest -q -rs --disable-warnings --tb=short -m 'not benchmark' -p no:cacheprovider $SUITES) > $D/c_suites.out 2>&1
echo "c suites rc $? $(tail -n 1 $D/c_suites.out) $(date +%T)" >> $L
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
(cd $NEW && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer python3 -m pytest -q --disable-warnings --tb=short -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec.py -m benchmark -k test_real_) > $D/c_tts_gate.out 2>&1
echo "c tts gate rc $? $(grep -E 'passed|failed' $D/c_tts_gate.out | tail -n 1) $(date +%T)" >> $L
(cd $NEW && CUDA_VISIBLE_DEVICES=2,3 QWEN3_OMNI_MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct python3 -m pytest -q --disable-warnings --tb=short -p no:cacheprovider tests/unit_test/qwen3_omni/test_code2wav_snake_beta.py -k test_real_) > $D/c_omni_gate.out 2>&1
echo "c omni gate rc $? $(grep -E 'passed|failed' $D/c_omni_gate.out | tail -n 1) $(date +%T)" >> $L
(cd $D && PYTHONPATH=$NEW python3 $HERE/pad_slot_gsan_probe.py padded --card 2) > $D/c_hash.out 2>&1
echo "c tts hash rc $? $(grep -o '"live_rows_sha256": "[0-9a-f]*"' $D/c_hash.out) $(date +%T)" >> $L
for arm in main new new main; do
  if [ $arm = main ]; then tree=$MAIN; else tree=$NEW; fi
  echo "c slot decode $arm $(cd $D && PYTHONPATH=$tree python3 $HERE/omni_slot_decode_bench.py --card 2 2>&1 | tail -n 1) $(date +%T)" >> $L
  echo "c snake bench $arm $(cd $D && PYTHONPATH=$tree python3 $HERE/snake_kernel_bench.py --card 2 2>&1 | tail -n 1) $(date +%T)" >> $L
done
python3 $HERE/omni_pair.py --label o5 --a-tree $MAIN --b-tree $NEW --a-card 2 --b-card 3 --port 20000 --passes s_c1_seeded,ns_c1_seeded,s_c16_seeded --score none > $D/o5.out 2>&1
echo "omni o5 rc $? $(date +%T)" >> $L
python3 $HERE/dump_identity.py o5_s_c1_seeded o5_ns_c1_seeded o5_s_c16_seeded > $D/c_identity.out 2>&1
echo "c identity rc $? $(tr '\n' ' ' < $D/c_identity.out) $(date +%T)" >> $L
echo "queue_omni3097c_done $(date +%T)" >> $L
