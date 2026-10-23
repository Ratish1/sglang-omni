#!/usr/bin/env bash
# Where #3097's 0.1 dB lies against the fp32 decode, by served segment (first frame from the tail graph, the ramp's
# eager-captured graphs, the compiled steady chunks): vocoder_truth_probe.py on main, D2 and D2 with the bias and
# residual sum rounded once (check branch), with the width-8 graphs compiled as served, then main and D2 with every
# graph captured eagerly; each pair of runs on cards 2 and 3 at once.
set -u
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
P=$T/tasks/qwen3-tts-perf/box/chain_20261022/vocoder_truth_probe.py
LIST=/data/prof_20261022/formal/meta_w2/long/L1_c16_run.lst
M=${M:?the main tree}
P2=${P2:?the D2 tree}
PC=${PC:?the check tree}
L=$D/queue.log
echo "queue truth2 start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d2 $(git -C $P2 rev-parse HEAD:sglang_omni) check $(git -C $PC rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
probe() {
  local name=$1 tree=$2 card=$3
  shift 3
  (cd $tree && TORCHINDUCTOR_CACHE_DIR=$D/inductor_truth2_$name PYTHONPATH=$tree timeout 1800 python3 $P run $LIST $D/truth5_$name.json --card $card "$@" > $D/truth5_$name.txt 2>&1)
  echo "truth5 $name rc $? $(date +%T)" >> $L
}
probe main_compiled $M 2 --compiled & probe d2_compiled $P2 3 --compiled & wait
probe check_compiled $PC 2 --compiled & probe main_eager $M 3 & wait
probe d2_eager $P2 2 & wait
for change in d2_compiled check_compiled; do
  (cd $M && PYTHONPATH=$M python3 $P compare $D/truth5_main_compiled.json $D/truth5_$change.json > $D/truth5_main_compiled_$change.txt 2>&1)
done
(cd $M && PYTHONPATH=$M python3 $P compare $D/truth5_main_eager.json $D/truth5_d2_eager.json > $D/truth5_main_eager_d2_eager.txt 2>&1)
(cd $M && PYTHONPATH=$M python3 $P compare $D/truth5_main_eager.json $D/truth5_main_compiled.json > $D/truth5_main_eager_main_compiled.txt 2>&1)
echo "queue_truth2_done $(date +%T)" >> $L
