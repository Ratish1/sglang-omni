#!/usr/bin/env bash
# 10-25 evening, new lease, cards 0 and 1: #3132's c1f SIM lean, which moves per reference clip (BRANCH_RESULTS 10-25).
# 1. p_cp1: main ab8193645 (t_mq) against #3132's head 2f6c524e1 (t_l6q), c1f (seed 1234) both orders, every server
#    under conditioning_probe, which names per request the first layer whose output differs.
# 2. p_l3: main + L3 (t_m3, scratch/m3 1847adb69) against the head + L3 (t_l63, scratch/l63 134841a02), from
#    $P/sim.bundle, the same L3 in both (the CUDA graph speech tokenizer, whose seeded audio repeats across boots),
#    c1f both orders under the probe, identity, WER and SIM.
# Decision rules, written before the run: (1) every diverged request differs first in the reference conditioning,
# which the PR does not compute, and every request with identical codes has byte identical audio; (2) with L3 in both
# arms nearly every request is byte identical and SIM moves by no more than the scorer's own noise (under 0.01 on
# the mean for identical audio); otherwise the vocoder change is the suspect.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025b/sim/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_sim.sh'
D=/data/c20261025b
P=$D/sim
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261025
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=44000,44400
export PAIR_PROBE_PATH=$HERE/conditioning_probe
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-speech-tokenizer-cuda-graphs
for pair in t_m3:scratch/m3 t_l63:scratch/l63; do
  name=${pair%%:*}
  git -C /workspace/sglang-omni fetch -q $P/sim.bundle ${pair#*:}
  [ -d /workspace/$name ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name FETCH_HEAD
done
for name in t_mq t_l6q t_m3 t_l63; do
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-60) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
probe_pair() {
  local first=$1 second=$2 name=$3
  (cd /workspace/sglang-omni && CONDITIONING_PROBE=$P/probe_$name PAIR_CELLS=c1f \
    python3 $IN/serve_pair_c20261021.py /workspace/$first /workspace/$second $P/$name > $P/$name.log 2>&1)
  log "pair $name rc $?"
  python3 $IN/profile_c20261022/identity.py $P/$name > $P/identity_$name.txt 2>&1
  log "identity $name $(tr -s ' ' < $P/identity_$name.txt | tr '\n' ' ' | cut -c1-300)"
  python3 $HERE/conditioning_probe_report.py $P/probe_$name ${first}_0.jsonl ${second}_1.jsonl \
    $P/$name/a/gpu0/c1f/audio $P/$name/a/gpu1/c1f/audio > $P/layers_$name.txt 2>&1
  python3 $HERE/conditioning_probe_report.py $P/probe_$name ${first}_1.jsonl ${second}_0.jsonl \
    $P/$name/b/gpu1/c1f/audio $P/$name/b/gpu0/c1f/audio >> $P/layers_$name.txt 2>&1
  sed 's/^/    /' $P/layers_$name.txt >> $P/queue.txt
}
probe_pair t_mq t_l6q p_cp1
probe_pair t_m3 t_l63 p_l3
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_mq && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
    > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_mq && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
score $P/p_l3/a/gpu0/c1f 0 36100 &
score $P/p_l3/a/gpu1/c1f 1 36400 &
wait
score $P/p_l3/b/gpu0/c1f 0 36700 &
score $P/p_l3/b/gpu1/c1f 1 37000 &
wait
log "queue sim done"
