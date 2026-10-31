set -u
# 10-31 the next perf item after the talker bisect: the thinker's decode payload carries the audio encoder's device
# audio_feature_lengths (merge.py prune_preprocessing_for_thinker, kept by project_thinker_to_decode), which 10-30 measured
# as a device copy inside the thinker's pack on its send loop (p90 1.9 ms, max 27 ms, 676 of 1,090 requests on main69).
# queue102's census of main466 showed no CUDA call in any pack, but its window repeats the warm pass's samples. This
# re-measures on upstream main 3cb3c6ccb (the tools branch's tree, sglang_omni identical) with shm_put_size_probe, one
# boot on card 0, SeedTTS EN streaming c64, a cold full-corpus pass and then a second pass over the same corpus.
log() { echo "$(date -u +%T) $*" >> /data/q105/queue.log; }
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
TREE=/data/ab/main3cb
R=/data/q105
PORT=29700
PIN0="numactl --physcpubind=0-15,64-79 --membind=0"
mkdir -p $R/put
cd $TREE || exit 1
log "queue105 start: tools $(git -C /data/tools rev-parse --short HEAD), main3cb $(git -C $TREE rev-parse --short HEAD) $(git -C $TREE rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C $TREE status --porcelain -- sglang_omni | wc -l) dirty"
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  log "port $PORT already in use"
  exit 1
fi
export SGLANG_OMNI_STRICT_PORT=1
setsid bash -c "echo \$\$ > $R/omni.pgid; exec env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TREE:$S/shm_put_size_probe OMNI_SHM_PUT_PROBE=$R/put \
  $PIN0 python3 -u -m sglang_omni.cli serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate \
  --preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768 --host 127.0.0.1 --port $PORT" > $R/omni.log 2>&1 &
healthy=0
for _ in $(seq 360); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then healthy=1; break; fi
  ps -o stat= -g "$(cat $R/omni.pgid 2>/dev/null)" 2>/dev/null | grep -qv '^Z' || break
  sleep 5
done
log "omni healthy=$healthy"
if [ $healthy = 1 ]; then
  for k in 1 2; do
    out=$R/pass$k
    mkdir -p $out
    echo "pass$k start $(date +%s.%N)" >> $R/passes.txt
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TREE $PIN0 python3 $S/run_bench.py gen --arm seedtts_en --port $PORT --concurrency 64 --out $out > $out/gen.log 2>&1
    rc=$?
    echo "pass$k end $(date +%s.%N)" >> $R/passes.txt
    log "pass$k gen rc $rc: $(tail -1 $out/gen.log | cut -c1-160)"
  done
else
  log "server not healthy"
fi
kill -TERM -- -"$(cat $R/omni.pgid)" 2>/dev/null
sleep 15
kill -0 -- -"$(cat $R/omni.pgid)" 2>/dev/null && kill -KILL -- -"$(cat $R/omni.pgid)"
for _ in $(seq 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 100 ] && break
  sleep 2
done
log "omni stopped, card 0 at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
log "queue105 done"
