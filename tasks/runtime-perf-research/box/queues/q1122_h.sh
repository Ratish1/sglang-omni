#!/usr/bin/env bash
# 11-22, node-0001, after q1122_g. (1) Seeded c1 wav identity: the index copy fix (IXC 4ca23252e) against main (NMAIN
# e0482a08c), and part 2 with both turn rules (OCVG5 46b5e1206) against its base (BVG 6aff09f46); card 0. (2) WER and SIM
# of q1122_g's speech alone c64 cells (gc1, gc4 against gc2, gc3), Qwen3-ASR on card 1, each SIM pass after card 1 is
# clear.
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain BVG=/workspace/bvg OCVG5=/workspace/ocvg5 IXC=/workspace/ixc
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167
ASR_PORT=30580
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
card1_used_mib() { nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
seeded() {
  local name=$1 tree=$2
  note $name $tree "seeded c1 card 0 port 28490"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28490 --out $O/$name \
    --loads 1 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes clean "--bench-extra=--seed 3451" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "^q1122_g done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py" > /dev/null; do sleep 20; done
log "q1122_h start"
(
  until [ "$(card1_used_mib)" -lt 1024 ]; do sleep 5; done
  cd $NMAIN
  setsid python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $ASR_PORT --asr.gpu 1 \
    > $O/asr_h_serve.log 2>&1 &
  echo $! > $O/asr_h.pgid
  until curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null || ! kill -0 $(cat $O/asr_h.pgid) 2>/dev/null; do
    sleep 5
  done
  for boot in gc1 gc2 gc3 gc4; do
    cell=$O/$boot/c64_events
    python3 -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --lang en --voice-clone --port $ASR_PORT \
      --asr-concurrency 64 --output-dir $cell/bench > $cell/transcribe.log 2>&1
    log "wer $boot rc $?"
  done
  kill -TERM -- -$(cat $O/asr_h.pgid) 2>/dev/null
  until [ "$(card1_used_mib)" -lt 1024 ]; do sleep 5; done
  for boot in gc1 gc2 gc3 gc4; do
    cell=$O/$boot/c64_events
    python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
      --output-dir $cell/bench > $cell/similarity.log 2>&1
    log "similarity $boot rc $? $(grep -m1 -o 'Tried to allocate [0-9.]* [GM]iB' $cell/similarity.log)"
  done
  python3 $K/wer_sim_cells.py rules_c64_1=$O/gc1/c64_events/bench base_c64_2=$O/gc2/c64_events/bench \
    base_c64_3=$O/gc3/c64_events/bench rules_c64_4=$O/gc4/c64_events/bench >> $O/queue.log 2>&1
  log "H2 read"
) &
scoring=$!
seeded hi1_b $IXC
seeded hi2_a $NMAIN
log "identity fix A $(python3 $K/wav_identity.py $O c1_clean hi2_a hi1_b 2>&1 | grep -m1 'shared')"
seeded hp1_b $OCVG5
seeded hp2_a $BVG
log "identity part 2 $(python3 $K/wav_identity.py $O c1_clean hp2_a hp1_b 2>&1 | grep -m1 'shared')"
log "H1 read"
wait $scoring
log "q1122_h done"
