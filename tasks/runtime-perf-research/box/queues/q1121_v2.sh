#!/usr/bin/env bash
# 11-21, node-0001, after q1121_both. The shared-memory change reworked to the receive side only (PSHM2 6ba47848c): a
# read over 1 MiB copies and allocates its host buffer off the stage event loop, smaller reads and every send keep main's
# path. The first form also moved sends off the loop and made speech sends on main wait for a relay credit held by video
# blocks (preprocess end to hop sent p50 50 against 2 to 12 ms). Units, the new test on main's sources (its large case
# must fail), mutants; then same-time pairs on both cards, swapped, colocated, speech 8/s seed 3451 beside video 1/s seed
# 7 (cap 8) unless noted:
#   V1  stack with both changes (NSBOTH2) against the stack (NSTACK), probes on with payload reads
#   V2  the shared-memory change alone (PSHM2) against main (NMAIN)
#   V3  both changes (PBOTH2) against main
#   V4  stack with both changes against main
#   V5  stack with both changes against main, speech 8/s beside the looping Video-MME client (4 in flight)
#   V6  both changes against main, speech alone c1 and c16
#   V7  the graph change alone (PFIX) against main, speech alone c16, main on card 0
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
U=$O/units_shm2
NMAIN=/workspace/nmain PFIX=/workspace/pfix NSTACK=/workspace/nstack
PSHM2=/workspace/pshm2 PBOTH2=/workspace/pboth2 NSBOTH2=/workspace/nsboth2
PSHM2_SHA=6ba47848c40385f21b1fb0f13a2b46fa6f5df599
PBOTH2_SHA=e654b5fb5005335625d5097f7d460a33d5c13dae
NSBOTH2_SHA=8eff23b7778a5b46cfe48d757acac94bd1ce0637
NMAIN_SHA=e0482a08cf48e6333952c974f3a5e3a4a3398361
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_PAYLOAD_READS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
mixed() {
  local name=$1 tree=$2 card=$3 offset=$4 probe=$5 video=$6
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  local probe_args=()
  if [ $probe = probe ]; then probe_args=(--probe --serve-env "$PROBES"); else :; fi
  local video_args=(--max-concurrency 8 --request-rate 1)
  if [ $video = loop ]; then video_args=(--max-concurrency 4); else :; fi
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video $video card $card $probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
    "${probe_args[@]}" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
        --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
        --max-tokens 256 --timeout-s 500 --warmup 0 "${video_args[@]}" \
        --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
      client=$!
      while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
      kill $client 2>/dev/null
      wait $client 2>/dev/null
      pass=$((pass + 1))
    done
  ) &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
alone() {
  local name=$1 tree=$2 card=$3 offset=$4 loads=$5
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts loads $loads alone card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pair_read() {
  local title=$1 cell=$2 base=$3 change=$4
  python3 $K/body_table.py $O $cell "$title" "$base" "$change" >> $O/queue.log 2>&1
}
until grep -q "q1121_both done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_v2 start"
mkdir -p $U
cd /workspace/sglang-omni
git fetch -q origin $PSHM2_SHA $PBOTH2_SHA $NSBOTH2_SHA
git worktree add -q --detach $PSHM2 $PSHM2_SHA
git worktree add -q --detach $PBOTH2 $PBOTH2_SHA
git worktree add -q --detach $NSBOTH2 $NSBOTH2_SHA
for tree in pshm2 pboth2 nsboth2; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni)"
done
SUITES="tests/unit_test/relay tests/unit_test/pipeline tests/unit_test/qwen3_tts/test_pipeline.py \
  tests/unit_test/audar_tts/test_pipeline.py tests/unit_test/moss_tts_local/test_pipeline.py \
  tests/unit_test/ming_omni/test_pipeline.py tests/unit_test/utils/test_ipc_weights_cuda.py"
cd $PSHM2
PYTHONPATH=$PSHM2 python3 -m pytest -q -p no:cacheprovider $SUITES > $U/pshm2_units.log 2>&1
log "units pshm2 rc $? $(grep -E 'passed|failed' $U/pshm2_units.log | tail -1)"
git worktree add -q --detach /workspace/prefix_shm2 $NMAIN_SHA
cd /workspace/prefix_shm2
git checkout -q $PSHM2_SHA -- tests/unit_test/relay/test_shm_relay.py
PYTHONPATH=/workspace/prefix_shm2 python3 -m pytest -q -p no:cacheprovider tests/unit_test/relay/test_shm_relay.py \
  > $U/prefix_new_tests.log 2>&1
log "new shm2 tests on main's sources rc $? $(grep -E 'passed|failed|error' $U/prefix_new_tests.log | tail -1)"
git checkout -q -- .
git worktree add -q --detach /workspace/smut2 $PSHM2_SHA
cd /workspace/smut2
T=tests/unit_test/relay/test_shm_relay.py::test_a_large_payload_read_copies_off_the_event_loop_and_a_small_one_on_it
for patch in shmread_large_copy_on_loop shmread_large_buffer_on_loop shmread_small_copy_off_loop; do
  git checkout -q -- .
  git apply $K/mutants/$patch.patch || { log "mutant $patch did not apply"; continue; }
  PYTHONPATH=/workspace/smut2 python3 -m pytest -q -p no:cacheprovider "$T" > $U/$patch.log 2>&1
  rc=$?
  log "mutant $patch rc $rc ($([ $rc = 0 ] && echo SURVIVED || echo killed)) $(grep -E 'passed|failed' $U/$patch.log | tail -1)"
done
git checkout -q -- .
quiet_host
both "mixed v1a_b0 $NSBOTH2 0 200 probe rate" "mixed v1a_s1 $NSTACK 1 200 probe rate"
quiet_host
both "mixed v1b_s0 $NSTACK 0 400 probe rate" "mixed v1b_b1 $NSBOTH2 1 400 probe rate"
pair_read "stack with both (receive-only shm) against the stack, speech 8/s and video 1/s" c64 stack=v1a_s1,v1b_s0 \
  both=v1a_b0,v1b_b1
python3 $K/mixed_cell_load.py $O v1a_b0 v1b_b1 v1a_s1 v1b_s0 >> $O/queue.log 2>&1
log "V1 read"
quiet_host
both "mixed v2a_x0 $PSHM2 0 600 noprobe rate" "mixed v2a_m1 $NMAIN 1 600 noprobe rate"
quiet_host
both "mixed v2b_m0 $NMAIN 0 800 noprobe rate" "mixed v2b_x1 $PSHM2 1 800 noprobe rate"
pair_read "receive-only shm against main, speech 8/s and video 1/s" c64 main=v2a_m1,v2b_m0 shm=v2a_x0,v2b_x1
python3 $K/mixed_cell_load.py $O v2a_x0 v2b_x1 v2a_m1 v2b_m0 >> $O/queue.log 2>&1
log "V2 read"
quiet_host
both "mixed v3a_b0 $PBOTH2 0 1000 noprobe rate" "mixed v3a_m1 $NMAIN 1 1000 noprobe rate"
quiet_host
both "mixed v3b_m0 $NMAIN 0 1200 noprobe rate" "mixed v3b_b1 $PBOTH2 1 1200 noprobe rate"
pair_read "both (receive-only shm) against main, speech 8/s and video 1/s" c64 main=v3a_m1,v3b_m0 both=v3a_b0,v3b_b1
python3 $K/mixed_cell_load.py $O v3a_b0 v3b_b1 v3a_m1 v3b_m0 >> $O/queue.log 2>&1
log "V3 read"
quiet_host
both "mixed v4a_b0 $NSBOTH2 0 1600 noprobe rate" "mixed v4a_m1 $NMAIN 1 1600 noprobe rate"
quiet_host
both "mixed v4b_m0 $NMAIN 0 1800 noprobe rate" "mixed v4b_b1 $NSBOTH2 1 1800 noprobe rate"
pair_read "stack with both against main, speech 8/s and video 1/s" c64 main=v4a_m1,v4b_m0 stack_both=v4a_b0,v4b_b1
python3 $K/mixed_cell_load.py $O v4a_b0 v4b_b1 v4a_m1 v4b_m0 >> $O/queue.log 2>&1
log "V4 read"
quiet_host
both "mixed v5a_b0 $NSBOTH2 0 2000 noprobe loop" "mixed v5a_m1 $NMAIN 1 2000 noprobe loop"
quiet_host
both "mixed v5b_m0 $NMAIN 0 2200 noprobe loop" "mixed v5b_b1 $NSBOTH2 1 2200 noprobe loop"
pair_read "stack with both against main, 8/s beside 4 looping videos" c64 main=v5a_m1,v5b_m0 stack_both=v5a_b0,v5b_b1
python3 $K/mixed_cell_load.py $O v5a_b0 v5b_b1 v5a_m1 v5b_m0 >> $O/queue.log 2>&1
log "V5 read"
quiet_host
both "alone v6a_b0 $PBOTH2 0 2400 1,16" "alone v6a_m1 $NMAIN 1 2400 1,16"
quiet_host
both "alone v6b_m0 $NMAIN 0 2600 1,16" "alone v6b_b1 $PBOTH2 1 2600 1,16"
for cell in c1 c16; do
  pair_read "both against main, speech alone $cell" $cell main=v6a_m1,v6b_m0 both=v6a_b0,v6b_b1
done
log "V6 read"
quiet_host
both "alone v7_m0 $NMAIN 0 200 16" "alone v7_p1 $PFIX 1 200 16"
pair_read "graph change against main, speech alone c16" c16 main=fe1_m1,v7_m0 graph=fe1_p0,v7_p1
log "q1121_v2 done"
