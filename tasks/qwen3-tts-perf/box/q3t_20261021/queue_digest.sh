#!/usr/bin/env bash
# V3's card-to-card difference, localized: the V3 PR head against itself on cards 2 and 3, seeded c1 over the full
# corpus, with every vocoder decode's input codes and output digested (decode_digest/sitecustomize.py), then main
# against itself the same way as the control. decode_digest_compare.py names the first decode whose output differs on
# equal input, with its runner mode and width.
set -u
export PAIR_ROOT=/data/q3t_20261021/pairs
D=/data/q3t_20261021
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
S=$T/tasks/omni_step_profiling_20260918/scripts
V3=/workspace/v3pr_98b0bbc14
M=/workspace/main_f31cf9e9a
L=$D/queue.log
mkdir -p $PAIR_ROOT
echo "queue digest start $(date +%T) v3 $(git -C $V3 rev-parse HEAD:sglang_omni) main $(git -C $M rev-parse HEAD:sglang_omni)" >> $L
for ARM in "v3:$V3" "main:$M"; do
  NAME=${ARM%%:*}; TREE=${ARM#*:}
  python3 $IN/pair.py --label ${NAME}_digest_c1 --tree $TREE --b-tree $TREE --concurrency 1 --a-card 2 --b-card 3 \
    --port 7000 --seed 1234 --decode-digest > $D/${NAME}_digest_c1.out 2>&1
  echo "${NAME}_digest_c1 rc $? $(date +%T) $(grep -h 'byte identical' $D/${NAME}_digest_c1.out)" >> $L
  python3 $S/decode_digest_compare.py $PAIR_ROOT/${NAME}_digest_c1/a/digest $PAIR_ROOT/${NAME}_digest_c1/b/digest \
    > $D/${NAME}_digest_compare.txt 2>&1
  echo "${NAME} digest compare rc $? $(date +%T)" >> $L
done
echo "queue_digest_done $(date +%T)" >> $L
