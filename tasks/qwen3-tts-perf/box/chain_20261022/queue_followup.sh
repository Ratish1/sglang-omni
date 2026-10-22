#!/usr/bin/env bash
# After the compiled decode keeps its adds for inductor: the vocoder identity probe of main against the new chain head
# (the compiled width-8 entries are expected bit identical now), the vocoder bench of the previous head against the new
# one, the codes-held-equal pair of main against D2 alone with the change (both orders), SeedTTS EN c16 and c64 pairs of
# main against the new head in both card orders, and the conv engine probe on the new head. After queue_runaway.sh.
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
M=/workspace/main_1b726ab64
C=${C:?the new chain head tree}
PREV=${PREV:?the previous chain head tree}
D2=${D2:?D2 alone with the change}
L=$D/queue.log
until grep -q queue_runaway_done $L; do sleep 20; done
echo "queue followup start $(date +%T) chain $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d2 $(git -C $D2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
PYTHONPATH=$M python3 $HERE/vocoder_identity_probe.py $D/identity_main.pt --card 3 > $D/identity_main.log 2>&1
PYTHONPATH=$C python3 $HERE/vocoder_identity_probe.py $D/identity_head.pt --card 3 > $D/identity_head.log 2>&1
PYTHONPATH=$C python3 $HERE/vocoder_identity_probe.py --compare $D/identity_main.pt $D/identity_head.pt > $D/identity_main_head.txt 2>&1
echo "identity main against head: $(tail -n 1 $D/identity_main_head.txt) $(date +%T)" >> $L
for arm in $C $PREV $PREV $C; do
  echo "== bench $arm" >> $D/bench_followup.log
  PYTHONPATH=$arm python3 $HERE/vocoder_decode_bench.py --card 3 2>&1 | grep -E "^(warm|tail) " >> $D/bench_followup.log
done
echo "bench done $(date +%T)" >> $L
run_pair() {
  local label=$1 a=$2 b=$3
  shift 3
  python3 $IN/pair.py --label $label --tree $a --b-tree $b "$@" > $D/$label.out 2>&1
  echo "$label rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log) $(grep 'byte identical' $D/$label.out)" >> $L
}
run_pair fins2_c1_a $M $D2 --concurrency 1 --a-card 2 --b-card 3 --port 7000 --seed 1234
run_pair fins2_c1_b $M $D2 --concurrency 1 --a-card 3 --b-card 2 --port 7600 --seed 1234
python3 $SEEDED/seeded_quality.py fins2_c1_a fins2_c1_b > $D/seeded_quality_d2_fix.txt 2>&1
run_pair fin2_c16_a $M $C --concurrency 16 --a-card 2 --b-card 3 --port 8200 --gpu-metrics
run_pair fin2_c16_b $M $C --concurrency 16 --a-card 3 --b-card 2 --port 8800 --gpu-metrics
run_pair fin2_c64_a $M $C --concurrency 64 --a-card 2 --b-card 3 --port 10000 --gpu-metrics
run_pair fin2_c64_b $M $C --concurrency 64 --a-card 3 --b-card 2 --port 10600 --gpu-metrics
labels=$(ls $PAIR_ROOT | grep "^fin2_c" | sort)
python3 $IN/pair_table.py $labels > $D/table_fin2.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_fin2.txt 2>&1
python3 $READ/first_audio_tail.py $labels > $D/first_audio_tail_fin2.txt 2>&1
PYTHONPATH=$C python3 $HERE/conv_engine_probe.py --card 3 > $D/conv_engines.txt 2>&1
echo "queue_followup_done $(date +%T)" >> $L
