#!/usr/bin/env bash
# 10-25, cards 0 and 1: p_l6q's c1f SIM leans to the head in both orders (-0.10 ; -0.05), all of it in the 69 ; 71
# requests whose tokens differ between the arms (flip set mean -1.52 ; -0.76), while 1019 ; 1017 byte identical
# requests score alike and b1f's flip set, where no prefix hop runs, reads -0.21 ; +0.79. New seeded draws bound it:
# pair p_l6s, main ab8193645 (t_mq) against the head 2f6c524e1 (t_l6q), c1f under seed 11 in order a and seed 12 in
# order b, then WER and SIM of each arm.
# Decision rule, written before the run: the head's c1f SIM stands if the new draws' flip set means are not both
# below -0.5 with the head lower overall in both; otherwise the lean is the PR's to explain.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025/l6q/queue_s.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_l6s.sh'
D=/data/c20261025
P=$D/l6q
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=44800,45200
log() { echo "$* $(date +%T)" >> $P/queue_s.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
(cd /workspace/sglang-omni && PAIR_CELLS=c1f_s11 PAIR_CELLS_B=c1f_s12 python3 $IN/serve_pair_c20261021.py \
  /workspace/t_mq /workspace/t_l6q $P/p_l6s > $P/p_l6s.log 2>&1)
log "pair p_l6s rc $?"
python3 $IN/profile_c20261022/identity.py $P/p_l6s > $P/identity_p_l6s.txt 2>&1
log "identity p_l6s $(tr -s ' ' < $P/identity_p_l6s.txt | tr '\n' ' ' | cut -c1-300)"
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
score $P/p_l6s/a/gpu0/c1f_s11 0 36100 &
score $P/p_l6s/a/gpu1/c1f_s11 1 36400 &
wait
score $P/p_l6s/b/gpu0/c1f_s12 0 36700 &
score $P/p_l6s/b/gpu1/c1f_s12 1 37000 &
wait
log "queue l6s done"
