#!/usr/bin/env bash
# 11-22, node-0001, card 1 after q1122_d. (1) SIM of the forced-split cells, which ran while the ASR server was still
# exiting (CUDA OOM at 364 MiB); each pass now waits for card 1 to be clear. (2) The step yields' mechanism read: the
# combined form with them (OCVG3 84eed4847) at speech alone c16 with the talker batch, thinker step and turn probes,
# against tb1 (the same form without them: the talker waited 174 ms a second, holds p99 22.7 ms).
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain OCVG3=/workspace/ocvg3
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS1=56-111,168-223
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
card1_used_mib() { nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
until grep -q "^q1122_d done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
until [ "$(card1_used_mib)" -lt 1024 ]; do sleep 5; done
log "q1122_e start"
cd $NMAIN
for boot in wf1_b wf2_a wf3_a wf4_b; do
  cell=$O/$boot/c64_clean
  python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
    --output-dir $cell/bench > $cell/similarity.log 2>&1
  log "similarity $boot rc $? $(grep -m1 -E 'OutOfMemory' $cell/similarity.log | cut -c1-80)"
done
python3 $K/wer_sim_cells.py comb_split1_1=$O/wf1_b/c64_clean/bench main_split1_2=$O/wf2_a/c64_clean/bench \
  main_split1_3=$O/wf3_a/c64_clean/bench comb_split1_4=$O/wf4_b/c64_clean/bench >> $O/queue.log 2>&1
log "E1 read"
name=tb2 port=31330
note $name $OCVG3 "seedtts c16 alone card 1 traced port $port"
taskset -c $CPUS1 python3 $K/baseline.py --tree $OCVG3 --model omni_colocated --card 1 --port $port --out $O/$name \
  --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events --probe --serve-env "$TRACE_PROBES" \
  > $O/$name.driver.log 2>&1
log "boot $name rc $?"
python3 $K/tail_first_window.py $O/tb2 --cell c16_events > $O/tb2.first_window.txt 2>&1
python3 $K/turn_waits.py $O/tb2 --cell c16_events > $O/tb2.turns_c16.txt 2>&1
python3 $K/turn_hold_parts.py $O/tb2 --cell c16_events > $O/tb2.holds_c16.txt 2>&1
grep -h "slowest\|all (median)" $O/tb2.first_window.txt >> $O/queue.log
grep -h "waiter priority 1" $O/tb2.turns_c16.txt >> $O/queue.log
grep -h "thinker holds handed" $O/tb2.holds_c16.txt >> $O/queue.log
log "E2 read"
log "q1122_e done"
