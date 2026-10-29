#!/usr/bin/env bash
# Block 18 of 10-29: D on its landing base. test/qwen3-tts-refenc-on-compile cee64ad96 is D merged onto the speedups
# branch (92025c654): the float32 encoder and the exact-length encode, both compiled per graph bucket. Its pair against
# the speedups branch is D's own cost where it would land. Card 0: the qwen3_tts suite on the merge, then the CI's ICL
# speaker-loss case drawn 300 times on it. Card 1: the served encode (encoder_speedups_gate.py --variants served).
# Then same-time pairs (serve_arm.py --barrier), SeedTTS EN 1.7B Base, both card orders, scored: non-streaming c16,
# streaming c16, non-streaming c1.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
K=$T/tasks/qwen3-tts-perf/box/next_20261029
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
IDS=common_voice_en_10119832-common_voice_en_10119840_icl_sentence
W=$D/block18
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs $W/barrier
git -C $REPO fetch -q origin test/qwen3-tts-refenc-on-compile
rm -rf /workspace/n29_refcomp
git -C $REPO worktree prune
git -C $REPO worktree add -q --detach /workspace/n29_refcomp cee64ad96
echo "queue block18 start $(date +%T) kit $(git -C $T rev-parse HEAD) refcomp $(git -C /workspace/n29_refcomp rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')compile $(git -C /workspace/n29_compile rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L

lane_zero() {
  (cd /workspace/n29_refcomp && CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $W/units_refcomp.txt 2>&1)
  echo "== units refcomp rc $?: $(grep -E '[0-9]+ passed|failed|error' $W/units_refcomp.txt | tail -n 1) $(date +%T)" >> $L
  local tree=/workspace/n29_refcomp
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $W/serve_draw.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port 19000 --model-path $M17 --tts_engine.gpu 0 --vocoder.gpu 0" > $W/serve_draw.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/serve_draw.pid) waited=0
  until curl -sf http://127.0.0.1:19000/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then echo "draw server not healthy" >> $L; kill -TERM -$pgid 2> /dev/null; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  (cd $tree && PYTHONPATH=$tree python3 $HERE/speaker_loss.py serve $D/block14/parity/cases.jsonl --ids $IDS \
    --base-url http://127.0.0.1:19000 --model $M17 --out $W/draw_refcomp --draws 300 --concurrency 16 >> $W/logs/draw.log 2>&1)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
  (cd $tree && PYTHONPATH=$tree python3 $HERE/speaker_loss.py score $W/draw_refcomp --card 0 > $W/summary_draw.txt 2>&1)
  echo "draw refcomp $(grep '^common_voice' $W/summary_draw.txt | cut -d'|' -f2-6) $(date +%T)" >> $L
}

lane_one() {
  (cd /workspace/n29_refcomp && PYTHONPATH=/workspace/n29_refcomp TORCHINDUCTOR_CACHE_DIR=$W/inductor_gate \
    python3 $K/encoder_speedups_gate.py $D/wavs --model $M17 --card 1 --variants served > $W/gate_refcomp.txt 2>&1)
  echo "gate refcomp rc $? $(grep -E '^== |official|base /' $W/gate_refcomp.txt | tr '\n' ';') $(date +%T)" >> $L
}

lane_zero &
zero=$!
lane_one &
one=$!
wait $zero $one

arm() {
  local barrier=$1 label=$2 tree=$3 card=$4 port=$5
  shift 5
  python3 $ARM --label $label --tree /workspace/n29_$tree --card $card --port $port --barrier $W/barrier/$barrier "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? startup $(cat $ARM_ROOT/$label/startup.txt 2> /dev/null) $(grep -h -E 'RTF mean|WER \(corpus|Max token hits|Throughput|TTFC mean' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(grep -h 'timed out' $W/logs/arm_$label.log) $(date +%T)" >> $L
}

pair() {
  local name=$1
  arm $name $2 &
  local a=$!
  arm $name $3 &
  local b=$!
  wait $a $b
}

N16="--concurrency 16 --no-stream --score"
S16="--concurrency 16 --score"
N1="--concurrency 1 --no-stream --score"
pair c16n_a "c16n_a_compile compile 0 11000 $N16" "c16n_a_refcomp refcomp 1 11500 $N16"
pair c16n_b "c16n_b_refcomp refcomp 0 12000 $N16" "c16n_b_compile compile 1 12500 $N16"
pair c16s_a "c16s_a_compile compile 0 13000 $S16" "c16s_a_refcomp refcomp 1 13500 $S16"
pair c16s_b "c16s_b_refcomp refcomp 0 14000 $S16" "c16s_b_compile compile 1 14500 $S16"
pair c1n_a "c1n_a_compile compile 0 17000 $N1" "c1n_a_refcomp refcomp 1 17500 $N1"
pair c1n_b "c1n_b_refcomp refcomp 0 18000 $N1" "c1n_b_compile compile 1 18500 $N1"
echo "queue block18 done $(date +%T)" >> $L
