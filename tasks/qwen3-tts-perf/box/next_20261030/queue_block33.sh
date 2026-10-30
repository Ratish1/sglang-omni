#!/usr/bin/env bash
# Block 33 of 10-30: the talker priority branch with the default-stream hand-off (perf/qwen3-tts-talker-priority-stream
# BR, main 466da750d merged). The qwen3_tts suite on its head, then the three hand-off tests called directly from the
# head's test file against three sglang_omni trees: the head; the previous design on the same merge (7927364c9: the
# event in the final payload, the vocoder's wait on it, chunks not ordered for the default stream); and main's.
# usage: D=... T=... BR=<commit> bash queue_block33.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BR=${BR:?the branch commit}
REPO=/workspace/sglang-omni
HEAD_TREE=/workspace/n30_prio_$BR
PREV=/workspace/n30_prefix_prev_$BR
PMAIN=/workspace/n30_prefix_main_$BR
W=$D/block33
L=$W/queue.log
mkdir -p $W
git -C $REPO fetch -q origin perf/qwen3-tts-talker-priority-stream
for tree in $HEAD_TREE $PREV $PMAIN; do
  git -C $REPO worktree add -q --detach $tree $BR
done
git -C $PREV checkout -q 7927364c9 -- sglang_omni
git -C $PMAIN checkout -q 466da750d -- sglang_omni
echo "queue block33 start $(date +%T) head $(git -C $HEAD_TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')prev sglang_omni $(git -C $REPO rev-parse 7927364c9:sglang_omni) main sglang_omni $(git -C $REPO rev-parse 466da750d:sglang_omni) kit $(git -C $T rev-parse HEAD)" >> $L

FILE=tests/unit_test/qwen3_tts/test_pipeline.py
TESTS="$FILE::test_qwen3_tts_vocoder_reads_final_codes_finished_on_the_talker_stream $FILE::test_qwen3_tts_default_stream_reads_stream_codes_finished_on_the_talker_stream $FILE::test_qwen3_tts_final_codes_payload_serializes_for_another_process"
for spec in "head $HEAD_TREE" "prev $PREV" "main $PMAIN"; do
  set -- $spec
  imported=$(cd $2 && PYTHONPATH=$2 python3 -c "import sglang_omni; print(sglang_omni.__file__)")
  (cd $2 && PYTHONPATH=$2 CUDA_VISIBLE_DEVICES=1 python3 -m pytest $TESTS -q -p no:cacheprovider -rA > $W/handoff_$1.txt 2>&1)
  echo "handoff $1 ($imported): $(grep -E '^(PASSED|FAILED|ERROR) ' $W/handoff_$1.txt | sed -E 's/ tests[^:]*::test_qwen3_tts_/ /' | cut -c1-110 | tr '\n' ';') $(tail -n 1 $W/handoff_$1.txt) $(date +%T)" >> $L
done
(cd $HEAD_TREE && PYTHONPATH=$HEAD_TREE CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $W/units_head.txt 2>&1)
echo "suite head: $(tail -n 1 $W/units_head.txt) $(date +%T)" >> $L
echo "queue block33 done $(date +%T)" >> $L
