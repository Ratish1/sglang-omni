#!/usr/bin/env bash
# After queue_branches.sh, whose x-vector arms exited at argument parsing: SeedTTS EN without reference text, non-streaming
# c16, 1.7B and 0.6B, main and the mask removal (B1); then seeded SeedTTS EN ICL c16 (seed 1234), main, the two-frame
# minimum (A) and the frame ceilings (C), each compared byte for byte with main: neither change can move a sentence's
# codes unless EOS competes in the first two frames or a request reaches its cap. Card 1 only.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
M=${M:?the main tree}
A=${A:?the two-frame minimum tree}
C=${C:?the frame ceilings tree}
B1=${B1:?the mask removal tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
export ARM_ROOT=$D/arms
CARD=1
L=$D/queue.log
until grep -q queue_branches_done $L; do
  sleep 30
done
echo "queue branches2 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
run_arm() {
  local label=$1 tree=$2
  shift 2
  python3 $ARM --label $label --tree $tree --card $CARD --port 21000 --no-stream --score "$@" >> $L 2>&1
  echo "arm $label rc $? $(date +%T)" >> $L
}
run_arm xvec17_c16_main $M --concurrency 16 --bench-args=--no-ref-text
run_arm xvec17_c16_B1 $B1 --concurrency 16 --bench-args=--no-ref-text
run_arm xvec06_c16_main $M --concurrency 16 --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --bench-args=--no-ref-text
run_arm xvec06_c16_B1 $B1 --concurrency 16 --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --bench-args=--no-ref-text
for arm in "main $M" "A $A" "C $C"; do
  set -- $arm
  python3 $ARM --label seeded_c16_$1 --tree $2 --card $CARD --port 21000 --no-stream --concurrency 16 --seed 1234 >> $L 2>&1
  echo "arm seeded_c16_$1 rc $? $(date +%T)" >> $L
done
for arm in A C; do
  echo "seeded c16 main against $arm: $(python3 $ARM --compare seeded_c16_main seeded_c16_$arm)" >> $L
done
echo "queue_branches2_done $(date +%T)" >> $L
