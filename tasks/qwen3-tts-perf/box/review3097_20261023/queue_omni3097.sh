#!/usr/bin/env bash
# The fused SnakeBeta in fp32 for every caller (Qwen3-Omni's code2wav joins Qwen3-TTS) and the slot stack without its
# eager fallback, against #3097's head: touched suites and the trained gate on the new head; the TTS padded live-row
# hash, which must equal the head's; Qwen3-Omni round 1 (head card 2, new card 3) with the fp32 probe over the seeded
# passes; a TTS seeded c1 pair for byte identity; Qwen3-Omni round 2 with the cards swapped and its probe.
set -u
D=/data/omni3097
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
OLD=${OLD:?the head tree}
NEW=${NEW:?the new tree}
HERE=$T/tasks/qwen3-tts-perf/box/review3097_20261023
TTS_KIT=$T/tasks/qwen3-tts-perf/box/v3_20261020
L=$D/queue.log
mkdir -p $PAIR_ROOT
echo "queue omni3097 start $(date +%T) old $(git -C $OLD rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')new $(git -C $NEW rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
SUITES="tests/unit_test/kernels/test_codec_state.py tests/unit_test/kernels/test_bias_snake_beta.py tests/unit_test/kernels/test_single_output_conv.py tests/unit_test/qwen3_tts/test_incremental_codec.py tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py tests/unit_test/qwen3_tts/test_pipeline.py tests/unit_test/qwen3_omni/test_code2wav.py tests/unit_test/qwen3_omni/test_code2wav_batching.py tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py tests/unit_test/qwen3_omni/test_code2wav_overlap.py tests/unit_test/qwen3_omni/test_code2wav_snake_beta.py tests/unit_test/qwen3_omni/test_codec_state_pool.py"
(cd $NEW && CUDA_VISIBLE_DEVICES=2,3 python3 -m pytest -q -rs --disable-warnings --tb=short -m 'not benchmark' -p no:cacheprovider $SUITES) > $D/suites_new.out 2>&1
echo "suites new rc $? $(tail -n 1 $D/suites_new.out) $(date +%T)" >> $L
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
(cd $NEW && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer python3 -m pytest -q --disable-warnings --tb=short -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec.py -m benchmark -k test_real_) > $D/gate_new.out 2>&1
echo "gate new rc $? $(tail -n 1 $D/gate_new.out) $(date +%T)" >> $L
(cd $D && PYTHONPATH=$NEW python3 $HERE/pad_slot_gsan_probe.py padded --card 2) > $D/hash_new_padded.out 2>&1
echo "tts hash new padded rc $? $(grep -o '"live_rows_sha256": "[0-9a-f]*"' $D/hash_new_padded.out) $(date +%T)" >> $L
python3 $HERE/omni_pair.py --label o1 --a-tree $OLD --b-tree $NEW --a-card 2 --b-card 3 --port 20000 > $D/o1.out 2>&1
echo "omni o1 rc $? $(date +%T)" >> $L
(cd $D && PYTHONPATH=$NEW python3 $HERE/omni_fp32_probe.py o1_s_c1_seeded o1_ns_c1_seeded --card 2) > $D/probe_o1.out 2>&1
echo "omni probe o1 rc $? $(date +%T)" >> $L
PAIR_ROOT=$PAIR_ROOT python3 $T/tasks/qwen3-tts-perf/box/v3_landing_20261022/seeded_quality.py o1_s_c1_seeded o1_ns_c1_seeded > $D/seeded_o1.txt 2>&1
echo "omni seeded quality o1 rc $? $(date +%T)" >> $L
PAIR_ROOT=$D/tts_pairs python3 $TTS_KIT/pair.py --label t_c1s --tree $OLD --b-tree $NEW --concurrency 1 --a-card 3 --b-card 2 --port 24000 --seed 1234 --skip-score > $D/t_c1s.out 2>&1
echo "tts c1 seeded rc $? $(grep 'byte identical' $D/t_c1s.out) $(date +%T)" >> $L
python3 $HERE/omni_pair.py --label o2 --a-tree $OLD --b-tree $NEW --a-card 3 --b-card 2 --port 22000 > $D/o2.out 2>&1
echo "omni o2 rc $? $(date +%T)" >> $L
(cd $D && PYTHONPATH=$NEW python3 $HERE/omni_fp32_probe.py o2_s_c1_seeded o2_ns_c1_seeded --card 2) > $D/probe_o2.out 2>&1
echo "omni probe o2 rc $? $(date +%T)" >> $L
PAIR_ROOT=$PAIR_ROOT python3 $T/tasks/qwen3-tts-perf/box/v3_landing_20261022/seeded_quality.py o1_s_c1_seeded o1_ns_c1_seeded o2_s_c1_seeded o2_ns_c1_seeded > $D/seeded_all.txt 2>&1
echo "queue_omni3097_done $(date +%T)" >> $L
