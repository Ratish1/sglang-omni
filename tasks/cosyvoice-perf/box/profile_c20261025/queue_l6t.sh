#!/usr/bin/env bash
# 10-25, cards 0 and 1: #3132's review F1 (tasks/pr3132_review_20261025): the prefill ladder drops L4's whole longest
# hop tiers, and on the review's H200 2 rows x 200 frames replay 16.55 ms at tier 416 against 14.17 ms at 400 (+2.4 ms
# for 16 padded frames, where the other shapes pay 12 to 22 us a frame). The PR head t_l6h (origin
# perf/cosyvoice3-prefix-graph-ladder-v2, 0b9be5746).
# Steps: on card 1 prefix_tier_profile.py's kernels, padding and full parts (one runner over every design's tiers);
# beside it on card 0 a census server (prefix_census/sitecustomize.py) through the full SeedTTS EN set at c16, then c64,
# streaming; then on card 1 the designs part over that census.
# Decision rules, written before the run: the kernels part names the kernel(s) whose time differs between tiers 400
# and 416 for the same rows; the census shows whether every step is a multiple of 50 frames; a tier design leaves the
# PR only if it prices at least 2 % under l6 on the sampled steps of both cells and no cell prices above l6.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261025/l6t/queue.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025/queue_l6t.sh'
D=/data/c20261025
P=$D/l6t
IN=/workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261025
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
PORT=43600
mkdir -p $P/census
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-prefix-graph-ladder-v2
sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
[ -d /workspace/t_l6h ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_l6h $sha
log "tree t_l6h $(git -C /workspace/t_l6h log --oneline -1 | cut -c1-60) sglang_omni $(git -C /workspace/t_l6h rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_l6h diff --quiet HEAD -- sglang_omni || echo DIRTY)"
profile() {
  (cd /workspace/t_l6h && CUDA_VISIBLE_DEVICES=$1 TORCHINDUCTOR_CACHE_DIR=$P/inductor_profile \
    PYTHONPATH=/workspace/t_l6h:$COSYVOICE taskset -c $(cpus $1) \
    timeout 3600 python3 $IN/prefix_tier_profile.py --out $P/$2 --parts $3 ${4:+--census $4} > $P/$2.log 2>&1
  log "profile $2 rc $?")
}
profile 1 profile kernels,padding,full &
cd /workspace/t_l6h
setsid env CUDA_VISIBLE_DEVICES=0 SGLANG_OMNI_STRICT_PORT=1 TORCHINDUCTOR_CACHE_DIR=$P/inductor_t_l6h \
  PREFIX_CENSUS=$P/census PYTHONPATH=$IN/prefix_census:/workspace/t_l6h:$COSYVOICE taskset -c $(cpus 0) \
  python3 -m sglang_omni.cli serve --model-path $MODEL --host 127.0.0.1 --port $PORT > $P/serve.log 2>&1 < /dev/null &
SERVE_PID=$!
log "serve pgid $SERVE_PID"
for i in $(seq 1 360); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  kill -0 $SERVE_PID 2> /dev/null || { log "server died"; break; }
  sleep 5
done
log "healthy, census lines at boot $(wc -l < $P/census/census.jsonl)"
for concurrency in 16 64; do
  echo "{\"cell\": \"c$concurrency\"}" >> $P/census/census.jsonl
  taskset -c $(cpus 0) env CUDA_VISIBLE_DEVICES= python3 -m benchmarks.eval.benchmark_tts_seedtts --model $MODEL \
    --port $PORT --lang en --use-existing-server --generate-only --stream --max-concurrency $concurrency \
    --output-dir $P/bench_c$concurrency > $P/bench_c$concurrency.log 2>&1
  log "bench c$concurrency rc $? $(grep -E 'Completed requests|Failed requests' $P/bench_c$concurrency.log | tr -s ' ' | tr '\n' ' ')"
done
kill -TERM -- -$SERVE_PID
sleep 20
kill -KILL -- -$SERVE_PID 2> /dev/null
log "census lines $(wc -l < $P/census/census.jsonl), tracebacks $(grep -c Traceback $P/serve.log), prefix tiers $(grep -o 'prefix graphs captured: tiers=[^]]*' $P/serve.log | head -1 | wc -c) chars"
wait
profile 1 designs designs $P/census/census.jsonl
log "queue l6t done"
