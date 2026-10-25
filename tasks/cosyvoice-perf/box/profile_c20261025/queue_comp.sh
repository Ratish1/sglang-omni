#!/usr/bin/env bash
# 10-25 evening, new lease, cards 0 and 1: #3132's SIM question at the component level (the user: compare the
# components' numbers against a reference implementation, low level, with solid infrastructure).
# 1. tokenizer_margins.py, three processes (a on card 0, b on card 1, c on card 0 after a): every distinct seed-tts EN
#    reference through main's ONNX Runtime tokenizer as served, ONNX Runtime's CPU provider, and the torch port of the
#    same graph (the L3 tree's speech_tokenizer.py) in float64, fp32 and TF32, with each token's float64 FSQ margin;
#    tokenizer_margins_report.py then joins the three runs and the served probe pair p_cp1 ($D/sim/probe_p_cp1).
# 2. prefix_tier_parity.py on card 1 after b: #3132's component, the Flow prefix hop at main's tiers, #3132's and
#    eagerly, bit for bit, each against upstream CosyVoice's DiT in fp32.
# Decision rules, written before the run: every token ONNX Runtime changes between processes or against float64 sits
# within TF32's rounding of the boundary (a float64 margin below 1e-3), and in every served divergence one or neither
# server holds the float64 codes, at a reference with such a margin; main's and #3132's tiers give bit identical mels,
# contexts and K and V in every step, and the same distance to fp32.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025b/comp/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_comp.sh'
D=/data/c20261025b
P=$D/comp
HERE=/workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
margins() {
  local label=$1 card=$2
  (cd /workspace/t_mq && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/t_mq:$COSYVOICE taskset -c $(cpus $card) \
    timeout 3600 python3 $HERE/tokenizer_margins.py --l3-tree /workspace/t_m3 --out $P/margins_$label.pt \
    > $P/margins_$label.log 2>&1)
  log "margins $label card $card rc $? $(grep -E 'against fp64' $P/margins_$label.log | tr '\n' ' ')"
}
parity() {
  (cd /workspace/t_l6q && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_l6q:$COSYVOICE \
    TORCHINDUCTOR_CACHE_DIR=$P/inductor_parity taskset -c $(cpus 1) \
    timeout 3600 python3 $HERE/prefix_tier_parity.py --main-tree /workspace/t_mq --out $P/parity.json \
    > $P/parity.log 2>&1)
  log "parity rc $? $(grep '^summary' $P/parity.log | cut -c1-600)"
}
(margins a 0; margins c 0) &
(margins b 1; parity) &
wait
python3 $HERE/tokenizer_margins_report.py $P/margins_a.pt $P/margins_b.pt $P/margins_c.pt \
  --probe $D/sim/probe_p_cp1 t_mq_0.jsonl t_l6q_1.jsonl t_mq_1.jsonl t_l6q_0.jsonl > $P/margins_report.txt 2>&1
log "margins report rc $?"
sed 's/^/    /' $P/margins_report.txt >> $P/queue.txt
log "queue comp done"
