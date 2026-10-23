#!/usr/bin/env bash
# #3097's review fix (padded graph rows skip the conv state exchanges) against the PR head it lands on. GSan through
# the arena wiring, three cases per tree on card 2, one process each; the padded case's live-row hash on both trees;
# the touched suites and the trained gate on the fix; the new TTS scratch-pool assertion on the PR head, which must
# fail; then one SeedTTS EN c16 pair per card order for speed, head as arm a and the fix as arm b.
set -u
export PAIR_ROOT=/data/review3097_20261023/pairs
D=/data/review3097_20261023
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
HERE=$T/tasks/qwen3-tts-perf/box/review3097_20261023
OLD=${OLD:?the PR head tree}
NEW=${NEW:?the fix tree}
PREFIX=${PREFIX:?the PR head tree carrying the TTS test file of the fix}
L=$D/queue.log
mkdir -p $PAIR_ROOT
BASES="20600 21800"
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
echo "queue review3097 start $(date +%T) old $(git -C $OLD rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')new $(git -C $NEW rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) busy ports [$BUSY]" >> $L
for tag in old new; do
  if [ $tag = old ]; then tree=$OLD; else tree=$NEW; fi
  for case in distinct scratch padded; do
    (cd $D && PYTHONPATH=$tree timeout 1800 python3 $HERE/pad_slot_gsan_probe.py $case --gsan --card 2) > $D/gsan_${tag}_${case}.out 2>&1
    rc=$?
    echo "gsan $tag $case rc $rc races $(grep -ci 'race detected' $D/gsan_${tag}_${case}.out) $(grep -o '"finished": true' $D/gsan_${tag}_${case}.out) $(date +%T)" >> $L
  done
  (cd $D && PYTHONPATH=$tree python3 $HERE/pad_slot_gsan_probe.py padded --card 2) > $D/hash_${tag}_padded.out 2>&1
  echo "hash $tag padded rc $? $(grep -o '"live_rows_sha256": "[0-9a-f]*"' $D/hash_${tag}_padded.out) $(date +%T)" >> $L
done
SUITES="tests/unit_test/kernels/test_codec_state.py tests/unit_test/kernels/test_bias_snake_beta.py tests/unit_test/kernels/test_single_output_conv.py tests/unit_test/qwen3_tts/test_incremental_codec.py tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py tests/unit_test/qwen3_tts/test_pipeline.py tests/unit_test/qwen3_omni/test_code2wav.py tests/unit_test/qwen3_omni/test_code2wav_batching.py tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py tests/unit_test/qwen3_omni/test_code2wav_overlap.py tests/unit_test/qwen3_omni/test_code2wav_snake_beta.py tests/unit_test/qwen3_omni/test_codec_state_pool.py"
(cd $NEW && CUDA_VISIBLE_DEVICES=2,3 python3 -m pytest -q -rs --disable-warnings --tb=short -m 'not benchmark' -p no:cacheprovider $SUITES) > $D/suites_new.out 2>&1
echo "suites new rc $? $(tail -n 1 $D/suites_new.out) $(date +%T)" >> $L
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
(cd $NEW && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer python3 -m pytest -q --disable-warnings --tb=short -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec.py -m benchmark -k test_real_) > $D/gate_new.out 2>&1
echo "gate new rc $? $(tail -n 1 $D/gate_new.out) $(date +%T)" >> $L
(cd $PREFIX && CUDA_VISIBLE_DEVICES=2,3 python3 -m pytest -q --disable-warnings --tb=short -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec.py -k test_arena_bound_graph_replays_match_eager_and_advance_the_arena) > $D/prefix_old.out 2>&1
echo "prefix old rc $? $(tail -n 1 $D/prefix_old.out) $(date +%T)" >> $L
if [ -n "$BUSY" ]; then
  echo "pairs skipped: listeners inside the pair port ranges" >> $L
  echo "queue_review3097_done $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  local label=$1
  shift
  python3 $IN/pair.py --label $label --tree $OLD --b-tree $NEW "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
}
run_pair r1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 20600
run_pair r1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 21800
python3 $IN/pair_table.py r1_c16_a r1_c16_b > $D/table_r1.txt 2>&1
python3 $READ/paired_by_prompt.py r1_c16_a r1_c16_b > $D/paired_r1.txt 2>&1
echo "queue_review3097_done $(date +%T)" >> $L
