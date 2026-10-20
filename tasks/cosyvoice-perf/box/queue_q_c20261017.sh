#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: the evidence of the graphs PR on the priority PR (item 1 with the priority, no ranking,
# t_gp_56b436580, against the priority alone on main, t_mp_ed979c580). Waits for queue_p. The kit profile of both at
# the same time, c16 and c64 of 640 unseeded requests traced, no py-spy, with each cell's first audio timeline and card
# busy span; then WER and SIM of p_gp's c16 audio, both arms in both orders, on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_q.pgid; exec bash queue_q_c20261017.sh'
D=/data/c20261017
IN=$D/in
G=t_gp_56b436580
P=t_mp_ed979c580
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
log() { echo "$* $(date +%T)" >> $D/queue_q.txt; }
score() {
  local dir=$1
  (cd /workspace/t_main2 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $?"
  (cd /workspace/t_main2 && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $?"
}

log "queue wait"
until grep -q "queue done" $D/queue_p.txt; do sleep 30; done
log "queue start"
mkdir -p $D/prof_gp
PEER=$D/prof_gp/graphs PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh $P 2 0-15,64-79 \
  21000 $D/prof_gp/base > $D/prof_gp/run_base.log 2>&1 &
first=$!
PEER=$D/prof_gp/base PYSPY=0 CELLS="c16:16:640 c64:64:640" bash $IN/cosy_profile_run_c20261017.sh $G 3 16-31,80-95 \
  22000 $D/prof_gp/graphs > $D/prof_gp/run_graphs.log 2>&1 &
second=$!
wait $first $second
log "profiles $(tail -n 1 $D/prof_gp/base/progress.txt) / $(tail -n 1 $D/prof_gp/graphs/progress.txt)"
for cell in c16 c64; do
  python3 $IN/first_audio_timeline_v2_c20261017.py base=$D/prof_gp/base/$cell.sqlite graphs=$D/prof_gp/graphs/$cell.sqlite \
    > $D/prof_gp/timeline_$cell.txt 2>&1
  python3 $IN/card_busy_span_c20261017.py base=$D/prof_gp/base/$cell.sqlite graphs=$D/prof_gp/graphs/$cell.sqlite \
    > $D/prof_gp/card_busy_$cell.txt 2>&1
  log "analysis $cell done"
done
for dir in p_gp/a/gpu0 p_gp/a/gpu1 p_gp/b/gpu0 p_gp/b/gpu1; do
  score $D/$dir/c16
done
log "queue done"
