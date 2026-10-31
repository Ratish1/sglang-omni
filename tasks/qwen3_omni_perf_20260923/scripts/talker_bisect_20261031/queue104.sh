set -u
# 10-31 the talker's bad outputs (runaways, babble from the first word, garbled starts), bisect step (b)/(c): upstream
# main 74003d00f plus the talker capture (test/qwen3-omni-talker-capture-20261031 7cb077f17, /data/ab/cap) serves
# SeedTTS EN voice clone, H100 BF16 colocated on card 0, streaming, c64, thinker 0.7, talker on its defaults, for
# PASSES full-corpus passes on one boot. Every talker request leaves its prefill rows, decode rows, per-step hidden,
# codes and batch sizes in /data/q104/capture. Each pass is scored (Qwen3-ASR at 64 in flight) on card 1 while the
# next pass runs. Wavs are kept for the bad requests' inspection.
log() { echo "$(date -u +%T) $*" >> /data/q104/queue.log; }
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
TREE=/data/ab/cap
R=/data/q104
PASSES=${PASSES:-8}
PORT=29400
ASR_PORT=29600
PIN0="numactl --physcpubind=0-15,64-79 --membind=0"
PIN1="numactl --physcpubind=16-31,80-95 --membind=0"
mkdir -p $R/capture
cd $TREE || exit 1
log "queue104 start: tools $(git -C /data/tools rev-parse --short HEAD), cap $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C $TREE status --porcelain -- sglang_omni | wc -l) dirty, $PASSES passes"

# serve <label> <card> <pin> <port> <model> <args...>: a server in its own process group, pgid in $R/<label>.pgid
serve() {
  local label=$1 card=$2 pin=$3 port=$4 model=$5; shift 5
  if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
    log "$label port $port already in use"
    return 1
  fi
  setsid bash -c "echo \$\$ > $R/$label.pgid; exec env CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$TREE \
    $pin python3 -u -m sglang_omni.cli serve --model-path $model $* --host 127.0.0.1 --port $port" > "$R/$label.log" 2>&1 &
  local began=$(date +%s) healthy=0
  for _ in $(seq 360); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; then healthy=1; break; fi
    ps -o stat= -g "$(cat "$R/$label.pgid" 2>/dev/null)" 2>/dev/null | grep -qv '^Z' || break
    sleep 5
  done
  log "$label healthy=$healthy startup_s $(( $(date +%s) - began ))"
  [ $healthy = 1 ]
}

stop() {
  local label=$1 card=$2
  kill -TERM -- -"$(cat "$R/$label.pgid")" 2>/dev/null
  sleep 15
  kill -0 -- -"$(cat "$R/$label.pgid")" 2>/dev/null && kill -KILL -- -"$(cat "$R/$label.pgid")"
  for _ in $(seq 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$card")" -lt 100 ] && break
    sleep 2
  done
  log "$label stopped, card $card at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$card")"
}

export SGLANG_OMNI_STRICT_PORT=1
serve asr 1 "$PIN1" $ASR_PORT Qwen/Qwen3-ASR-1.7B || exit 1
SGLANG_OMNI_TALKER_CAPTURE_DIR=$R/capture serve omni 0 "$PIN0" $PORT Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 || { stop asr 1; exit 1; }
scores=""
for k in $(seq $PASSES); do
  out=$R/pass$k
  mkdir -p $out
  echo "pass$k start $(date +%s.%N)" >> $R/passes.txt
  (cd $TREE && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TREE $PIN0 python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 64 --out $out > $out/gen.log 2>&1)
  rc=$?
  echo "pass$k end $(date +%s.%N)" >> $R/passes.txt
  log "pass$k gen rc $rc: $(tail -1 $out/gen.log | cut -c1-200), captures $(ls $R/capture | wc -l)"
  (cd $TREE && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$TREE $PIN1 python3 $S/run_bench.py score --arm seedtts_en --asr-port $ASR_PORT --out $out > $out/score.log 2>&1; \
    log "pass$k score rc $?: $(tr -d '\n ' < $out/seedtts_en/score_summary.json | cut -c1-300)") &
  scores="$scores $!"
done
stop omni 0
wait $scores
stop asr 1
log "queue104 done"
