#!/usr/bin/env bash
# 11-21, node-0001, after q1121_payload. The second change, shared-memory copies off the stage event loop (PSHM
# 21d117e5d on main e0482a08c): its units, the new tests on main's sources (they must fail), each guarantee reverted
# alone. Then both changes against their bases at equal arrivals of speech and video (no closed-loop video client, whose
# faster arm draws more video work): speech 8/s seed 3451 and Video-MME 1/s seed 7 (cap 8), colocated, same-time pairs
# on both cards, swapped:
#   E1  stack with both changes (NSBOTH) against the stack (NSTACK), probes on with payload reads
#   E2  both changes (PBOTH) against main (NMAIN)
#   E3  the shared-memory change alone (PSHM) against main
#   E4  the graph prefill change alone (PFIX) against main
#   E5  both changes against main, speech alone c1 and c16 (every request crosses the changed transport)
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
U=$O/units_shm
NMAIN=/workspace/nmain PFIX=/workspace/pfix NSTACK=/workspace/nstack
PSHM=/workspace/pshm PBOTH=/workspace/pboth NSBOTH=/workspace/nsboth
PSHM_SHA=21d117e5d42620a0c075dc970f7160e4295f32e5
PBOTH_SHA=8d94be4d424f736f8a0a6e96c7e42b9c06215709
NSBOTH_SHA=054cea3237cef4847688717ac3ddae50c12f60e3
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
  local name=$1 tree=$2 card=$3 offset=$4 probe=$5
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  local probe_args=()
  if [ $probe = probe ]; then probe_args=(--probe --serve-env "$PROBES"); else :; fi
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s seed $OMNI_VIDEO_ARRIVAL_SEED cap 8 card $card $probe"
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
        --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 --request-rate 1 \
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
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts c1 and c16 alone card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 1,16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1121_payload done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_both start"
mkdir -p $U
cd /workspace/sglang-omni
git fetch -q origin $PSHM_SHA $PBOTH_SHA $NSBOTH_SHA
git worktree add -q --detach $PSHM $PSHM_SHA
git worktree add -q --detach $PBOTH $PBOTH_SHA
git worktree add -q --detach $NSBOTH $NSBOTH_SHA
for tree in pshm pboth nsboth; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni)"
done
SUITES="tests/unit_test/relay tests/unit_test/pipeline tests/unit_test/qwen3_tts/test_pipeline.py \
  tests/unit_test/audar_tts/test_pipeline.py tests/unit_test/moss_tts_local/test_pipeline.py \
  tests/unit_test/ming_omni/test_pipeline.py tests/unit_test/utils/test_ipc_weights_cuda.py"
cd $PSHM
PYTHONPATH=$PSHM python3 -m pytest -q -p no:cacheprovider $SUITES > $U/pshm_units.log 2>&1
log "units pshm rc $? $(grep -E 'passed|failed' $U/pshm_units.log | tail -1)"
git worktree add -q --detach /workspace/prefix_shm $NMAIN_SHA
cd /workspace/prefix_shm
git checkout -q $PSHM_SHA -- tests/unit_test/relay/test_shm_relay.py
PYTHONPATH=/workspace/prefix_shm python3 -m pytest -q -p no:cacheprovider tests/unit_test/relay/test_shm_relay.py \
  > $U/prefix_new_tests.log 2>&1
log "new shm tests on main's sources rc $? $(grep -E 'passed|failed' $U/prefix_new_tests.log | tail -1)"
git worktree add -q --detach /workspace/smut $PSHM_SHA
cd /workspace/smut
T=tests/unit_test/relay/test_shm_relay.py
while read -r patch test; do
  git checkout -q -- .
  git apply $K/mutants/$patch.patch || { log "mutant $patch did not apply"; continue; }
  PYTHONPATH=/workspace/smut python3 -m pytest -q -p no:cacheprovider "$T::$test" > $U/$patch.log 2>&1
  rc=$?
  log "mutant $patch rc $rc ($([ $rc = 0 ] && echo SURVIVED || echo killed)) $(grep -E 'passed|failed' $U/$patch.log | tail -1)"
done <<'EOF'
shmloop_get_on_loop test_shm_copies_leave_the_event_loop_running
shmloop_put_on_loop test_shm_copies_leave_the_event_loop_running
shmloop_cancelled_put_leaks test_a_cancelled_shm_put_unlinks_its_block_and_returns_its_credit
shmloop_buffer_on_loop test_shm_payload_read_allocates_its_host_buffer_off_the_event_loop
EOF
git checkout -q -- .
quiet_host
both "mixed g1_b0 $NSBOTH 0 200 probe" "mixed g1_s1 $NSTACK 1 200 probe"
quiet_host
both "mixed g2_s0 $NSTACK 0 400 probe" "mixed g2_b1 $NSBOTH 1 400 probe"
python3 $K/body_table.py $O c64 "stack with both changes against the stack, speech 8/s and video 1/s" stack=g1_s1,g2_s0 \
  both=g1_b0,g2_b1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O g1_b0 g2_b1 g1_s1 g2_s0 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O stack=g1_s1,g2_s0 both=g1_b0,g2_b1 > $O/both_first_gap_stack.txt 2>&1
log "E1 read"
quiet_host
both "mixed h1_b0 $PBOTH 0 600 noprobe" "mixed h1_m1 $NMAIN 1 600 noprobe"
quiet_host
both "mixed h2_m0 $NMAIN 0 800 noprobe" "mixed h2_b1 $PBOTH 1 800 noprobe"
python3 $K/body_table.py $O c64 "both changes against main, speech 8/s and video 1/s" main=h1_m1,h2_m0 both=h1_b0,h2_b1 \
  >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O h1_b0 h2_b1 h1_m1 h2_m0 >> $O/queue.log 2>&1
log "E2 read"
quiet_host
both "mixed i1_x0 $PSHM 0 1000 noprobe" "mixed i1_m1 $NMAIN 1 1000 noprobe"
quiet_host
both "mixed i2_m0 $NMAIN 0 1200 noprobe" "mixed i2_x1 $PSHM 1 1200 noprobe"
python3 $K/body_table.py $O c64 "shared-memory change against main, speech 8/s and video 1/s" main=i1_m1,i2_m0 \
  shm=i1_x0,i2_x1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O i1_x0 i2_x1 i1_m1 i2_m0 >> $O/queue.log 2>&1
log "E3 read"
quiet_host
both "mixed j1_p0 $PFIX 0 1600 noprobe" "mixed j1_m1 $NMAIN 1 1600 noprobe"
quiet_host
both "mixed j2_m0 $NMAIN 0 1800 noprobe" "mixed j2_p1 $PFIX 1 1800 noprobe"
python3 $K/body_table.py $O c64 "graph prefill change against main, speech 8/s and video 1/s" main=j1_m1,j2_m0 \
  graph=j1_p0,j2_p1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O j1_p0 j2_p1 j1_m1 j2_m0 >> $O/queue.log 2>&1
log "E4 read"
quiet_host
both "alone k1_b0 $PBOTH 0 2000" "alone k1_m1 $NMAIN 1 2000"
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "both changes against main, speech alone $cell" main=k1_m1 both=k1_b0 >> $O/queue.log 2>&1
done
log "q1121_both done"
