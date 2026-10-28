#!/usr/bin/env bash
# Block 6 of 10-28: which mechanism made block 5's customvoice1 server answer 1,799 requests with
# HTTP 500 "cuDNN SDPA execution failed ... CUDA ran out of memory outside PyTorch's allocator".
# The same 10,871 CustomVoice ZH cases (block 5's part01) on four main servers at once, seeded,
# non-streaming, c32: cards 0 and 1 as main runs, cards 2 and 3 with torch's cuDNN SDPA backend
# off (probe_site/sitecustomize.py). Every server process logs its PyTorch reserved memory and the
# device's free memory each second, and nvidia-smi logs each card's used memory each second.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
SCAN=$T/tasks/qwen3-tts-perf/box/review_20261026/budget_scan.py
PROBE=$T/tasks/qwen3-tts-perf/box/next_20261028/probe_site
MAIN=/workspace/n28_main
CV=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
CASES=$D/wide/cases/customvoice_part01.jsonl
W=$D/cudnn
L=$W/queue.log
mkdir -p $W
echo "queue block6 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) cases $(wc -l < $CASES)" >> $L
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader -lms 1000 > $W/gpu_memory.csv 2>&1 &
SMI=$!

scan_one() {
  local label=$1 card=$2 port=$3 cudnn_off=$4
  mkdir -p $W/memory_$label
  (cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$PROBE:$MAIN OMNI_PROBE_MEMORY_DIR=$W/memory_$label OMNI_PROBE_CUDNN_SDP_OFF=$cudnn_off setsid bash -c "echo \$\$ > $W/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path $CV --tts_engine.gpu $card --vocoder.gpu $card" > $W/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then echo "$label not healthy" >> $L; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
  (cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN scan $CASES --base-url http://127.0.0.1:$port --model $CV --label customvoice \
    --out $W/$label.jsonl --concurrency 32 > $W/$label.summary.txt 2>&1)
  echo "$label scanned rc $? errors $(grep -c '"error"' $W/$label.jsonl) of $(wc -l < $W/$label.jsonl) $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
pids=()
scan_one cudnn_on0 0 6100 0 & pids+=($!)
scan_one cudnn_on1 1 6600 0 & pids+=($!)
scan_one cudnn_off2 2 7100 1 & pids+=($!)
scan_one cudnn_off3 3 7600 1 & pids+=($!)
wait ${pids[@]}
kill $SMI 2> /dev/null
echo "queue block6 done $(date +%T)" >> $L
