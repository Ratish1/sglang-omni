#!/usr/bin/env bash
# 11-22, node-0001, card 1 after q1122_a. Seeded c64 below-50 % WER: main 1.651 / 1.683, part 2's base 1.679 / 1.668,
# combined (one context) 1.760 / 1.708; seeded c1 is byte identical across them, and at c64 even two boots of one tree
# keep 8 to 37 of 1,088 wavs identical (batch composition moves the rounding). Candidate layer: FlashAttention's KV
# split count, which follows each batch's total blocks. Measurement arm: every FA3 backend forced to one split
# (OMNI_HOP_PROBE_FA_NUM_SPLITS=1), seeded c64, combined (OCVG 7cae415a1) and main (NMAIN e0482a08c), B A A B, then
# WER and SIM on card 1. If the arms meet, the split rounding is the moving layer.
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain OCVG=/workspace/ocvg
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS1=56-111,168-223
ASR_PORT=30540
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
split_one() {
  local name=$1 tree=$2 port=$3
  note $name $tree "seeded c64 alone card 1 one FA split port $port"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes clean "--bench-extra=--seed 3451" \
    --probe --serve-env "OMNI_HOP_PROBE_FA_NUM_SPLITS=1 OMNI_HOP_PROBE_QUEUES=0" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "^q1122_a done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
log "q1122_d start"
split_one wf1_b $OCVG 31370
split_one wf2_a $NMAIN 31370
split_one wf3_a $NMAIN 31370
split_one wf4_b $OCVG 31370
grep -c "hop_probe patch" $O/wf1_b/serve.log | xargs -I{} log "wf1_b probe patch errors {}"
cd $NMAIN
setsid python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $ASR_PORT --asr.gpu 1 \
  > $O/asr_d_serve.log 2>&1 &
echo $! > $O/asr_d.pgid
until curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null || ! kill -0 $(cat $O/asr_d.pgid) 2>/dev/null; do
  sleep 5
done
for boot in wf1_b wf2_a wf3_a wf4_b; do
  cell=$O/$boot/c64_clean
  python3 -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --lang en --voice-clone --port $ASR_PORT \
    --asr-concurrency 64 --output-dir $cell/bench > $cell/transcribe.log 2>&1
  log "wer $boot rc $?"
done
kill -TERM -- -$(cat $O/asr_d.pgid) 2>/dev/null
for boot in wf1_b wf2_a wf3_a wf4_b; do
  cell=$O/$boot/c64_clean
  python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
    --output-dir $cell/bench > $cell/similarity.log 2>&1
  log "similarity $boot rc $? $(grep -m1 -E 'OutOfMemory' $cell/similarity.log | cut -c1-80)"
done
python3 $K/wer_sim_cells.py comb_split1_1=$O/wf1_b/c64_clean/bench main_split1_2=$O/wf2_a/c64_clean/bench \
  main_split1_3=$O/wf3_a/c64_clean/bench comb_split1_4=$O/wf4_b/c64_clean/bench >> $O/queue.log 2>&1
for pair in "wf2_a wf3_a" "wf1_b wf4_b" "wf2_a wf1_b"; do
  set -- $pair
  log "identity $1 $2 $(python3 $K/wav_identity.py $O c64_clean $1 $2 2>&1 | grep -m1 'shared')"
done
log "D read"
log "q1122_d done"
