#!/usr/bin/env bash
# 11-21, node-0001, card 0 after q1121_oc3 (WER on card 1 after q1121_vg). The combined form against main: both changes,
# part 1 (every thinker prefill chunk on the graph, deepstack rows included) and part 2 (thinker, talker and code2wav in
# one process and context, the talker first on the card): OCVG 85ba60a52, tree e9ab4004b74c. Card 0 alone, B A A B,
# every video boot gated:
#   F1  speech 8/s seed 3451 beside video 1/s seed 7 (cap 8), warm pass (every video new) and events pass
#   F2  speech alone c1, c16 and c64
#   F3  seeded c1 identity (seed 3451), byte identity of every wav against main
#   F4  WER and SIM of F2's c16 and c64 cells and F1's events cells, Qwen3-ASR on card 1
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain OCVG=/workspace/ocvg
OCVG_SHA=85ba60a52
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=0-55,112-167
ASR_PORT=30500
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
BOOTS=0
LAST=""
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
next_port() {
  BOOTS=$((BOOTS + 1))
  echo $((28000 + 150 + 200 * (BOOTS % 20)))
}
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
}
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 --request-rate 1 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
mixed_once() {
  local name=$1 tree=$2
  local port=$(next_port)
  quiet_host
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s card 0 port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $port $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
mixed() {
  local name=$1 tree=$2
  mixed_once $name $tree
  LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  if [ -n "$p50" ] && [ "$p50" -le 649 ]; then
    log "$line, within 649"
  else
    log "$line, over 649: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree
    LAST=${name}r
    log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O ${name}r)"
  fi
}
alone() {
  local name=$1 tree=$2 loads=$3
  local port=$(next_port)
  quiet_host
  note $name $tree "seedtts loads $loads alone card 0 port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
seeded() {
  local name=$1 tree=$2
  local port=$(next_port)
  quiet_host
  note $name $tree "seeded c1 card 0 port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1121_oc3 done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-one-context-visual-graph
git worktree add -q --detach $OCVG FETCH_HEAD
log "tree ocvg $(git -C $OCVG rev-parse --short HEAD) $(git -C $OCVG rev-parse --short=12 HEAD:sglang_omni)"
if [ "$(git -C $OCVG rev-parse --short=9 HEAD)" != "$OCVG_SHA" ]; then
  log "q1121_final2 stopped: ocvg is not $OCVG_SHA"
  exit 1
else :; fi
log "q1121_final2 start, $(node_free)"
seeded fi1_b $OCVG
seeded fi2_a $NMAIN
log "identity $(python3 $K/wav_identity.py $O c1_clean fi2_a fi1_b 2>&1 | tail -1)"
mixed fm1_b $OCVG
b1=$LAST
mixed fm2_a $NMAIN
a2=$LAST
mixed fm3_a $NMAIN
a3=$LAST
mixed fm4_b $OCVG
b4=$LAST
for cell in c64; do
  python3 $K/body_table.py $O $cell "combined (both, part 1, part 2) against main, card 0 B A A B, speech 8/s and video 1/s" \
    main=$a2,$a3 combined=$b1,$b4 >> $O/queue.log 2>&1
done
python3 $K/first_chunk_gap.py --cell c64_events $O main=$a2,$a3 combined=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/first_chunk_gap.py --cell c64_warm $O main=$a2,$a3 combined=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "F1 read"
alone fs1_b $OCVG 1,16,64
alone fs2_a $NMAIN 1,16,64
alone fs3_a $NMAIN 1,16,64
alone fs4_b $OCVG 1,16,64
for cell in c1 c16 c64; do
  python3 $K/body_table.py $O $cell "combined against main, card 0 B A A B, speech alone $cell" \
    main=fs2_a,fs3_a combined=fs1_b,fs4_b >> $O/queue.log 2>&1
done
log "F2 read"
until grep -q "q1121_vg done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
CELLS=""
for boot in fs1_b fs2_a fs3_a fs4_b; do
  CELLS="$CELLS $O/$boot/c16_events $O/$boot/c64_events"
done
cd $NMAIN
setsid python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $ASR_PORT --asr.gpu 1 \
  > $O/asr_final2_serve.log 2>&1 &
echo $! > $O/asr_final2.pgid
until curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null || ! kill -0 $(cat $O/asr_final2.pgid) 2>/dev/null; do
  sleep 5
done
log "asr server up $(curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null && echo yes || echo no)"
(
  for cell in $CELLS; do
    python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
      --output-dir $cell/bench > $cell/similarity.log 2>&1
    log "similarity $cell rc $?"
  done
) &
similarity_loop=$!
for cell in $CELLS; do
  python3 -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --lang en --voice-clone --port $ASR_PORT \
    --asr-concurrency 64 --output-dir $cell/bench > $cell/transcribe.log 2>&1
  log "wer $cell rc $?"
done
wait $similarity_loop
kill -TERM -- -$(cat $O/asr_final2.pgid) 2>/dev/null
python3 $K/wer_sim_cells.py comb_c16_1=$O/fs1_b/c16_events/bench comb_c16_4=$O/fs4_b/c16_events/bench \
  main_c16_2=$O/fs2_a/c16_events/bench main_c16_3=$O/fs3_a/c16_events/bench \
  comb_c64_1=$O/fs1_b/c64_events/bench comb_c64_4=$O/fs4_b/c64_events/bench \
  main_c64_2=$O/fs2_a/c64_events/bench main_c64_3=$O/fs3_a/c64_events/bench >> $O/queue.log 2>&1
log "F4 read"
log "q1121_final2 done"
