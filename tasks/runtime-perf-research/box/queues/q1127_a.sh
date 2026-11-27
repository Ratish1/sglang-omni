#!/usr/bin/env bash
# 11-27, node-0000, one H200 (host index 1, NUMA 1, CPUs 56-111,168-223; the container sees it as GPU 0). #3452's review
# fixes (FIX 72be1f80e: main merged into the PR head 4b1d6b482, then the stop path, the limit's name, the CPU bounds, docs)
# against upstream main (MAIN 0f70d9c7c), with the PR head (HEAD) where the stop path is the question. One card, so the
# arms alternate boots in mirrored order instead of same-time pairs.
#   1. CPU: the affected suites on FIX, the three new tests against their mutants, payload identity against main's
#      serial hashes on 112 CPUs and on 16, the thread census on 16 CPUs for HEAD and FIX.
#   2. Stop under video load (Ctrl-C to the group, 16 videos in flight): M H F F H M.
#   3. CI gates, 6 passes a boot, M F F M: text Video-MME c16 and the talker test on the CI speech pipeline, then
#      Video-AMME c8 (the load of one of the CI's two replicas) on the FP8 colocated server.
#   4. Equal offered load, colocated: speech 8/s (seed 3451, cap 64) beside video 1/s (seed 7, 8 at most), M F F M.
#   5. A 16-CPU host, the same cell with the server alone under taskset 56-63,168-175, the clients on the socket's
#      other CPUs: M F.
set -u
O=/data/v1127
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main HEAD=/workspace/head FIX=/workspace/fix
CPUS=56-111,168-223
SMALL=56-63,168-175
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
}
tree_of() { case $1 in m) echo $MAIN ;; h) echo $HEAD ;; f) echo $FIX ;; *) exit 2 ;; esac; }

until grep -q "setup done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1127_a start"

cd $FIX
PYTHONPATH=$FIX taskset -c $CPUS python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni \
  tests/unit_test/preprocessing tests/unit_test/minicpm_o tests/unit_test/ming_omni > $O/units_fix.log 2>&1
log "units fix rc $? $(grep -E '[0-9]+ (passed|failed)' $O/units_fix.log | tail -1)"
git -C /workspace/sglang-omni worktree add -q --detach /workspace/mut $(git -C $FIX rev-parse HEAD)
T=/workspace/mut
for spec in \
  "fix_stop_cancels_pool_work tests/unit_test/qwen3_omni/test_pipeline.py::test_preprocessing_stop_lets_a_running_video_finish_its_queued_frame_work" \
  "fix_videos_past_the_cpus tests/unit_test/qwen3_omni/test_pipeline.py::test_preprocessing_runs_no_more_videos_than_its_limit_or_the_stage_cpus" \
  "fix_reader_past_its_share tests/unit_test/preprocessing/test_video.py::test_a_frame_pool_video_opens_its_reader_on_its_share_of_the_decode_threads"; do
  set -- $spec
  git -C $T checkout -q -- .
  (cd $T && git apply $K/mutants/$1.patch && PYTHONPATH=$T timeout 300 python3 -m pytest -q -p no:cacheprovider "$2") \
    > $O/mutant_$1.log 2>&1
  rc=$?
  (cd $FIX && PYTHONPATH=$FIX python3 -m pytest -q -p no:cacheprovider "$2") > $O/mutant_$1.fix.log 2>&1
  frc=$?
  log "mutant $1 rc $rc ($([ $rc = 0 ] && echo SURVIVED || echo killed)), fix rc $frc $(tail -1 $O/mutant_$1.fix.log)"
done
git -C $T checkout -q -- .

run() {
  local tree=$1 cpus=$2 threads=$3; shift 3
  PYTHONPATH=$tree OMP_NUM_THREADS=$threads taskset -c $cpus python3 "$@"
}
run $MAIN $CPUS 112 $K/preprocess_shared_identity.py --threads 1 --image $O/id.png --out $O/id_main.json > $O/id_main.log 2>&1
log "identity main rc $? $(grep -E '^serial' $O/id_main.log | head -1)"
run $FIX $CPUS 112 $K/preprocess_shared_identity.py --threads 8 --image $O/id.png --out $O/id_fix.json \
  --against $O/id_main.json > $O/id_fix.log 2>&1
log "identity fix 112 cpus rc $? $(grep -E 'differ' $O/id_fix.log | sed 's/.*: //' | tr '\n' ' ')"
run $FIX $SMALL 8 $K/preprocess_shared_identity.py --threads 8 --image $O/id.png --out $O/id_fix16.json \
  --against $O/id_main.json > $O/id_fix16.log 2>&1
log "identity fix 16 cpus rc $? $(grep -E 'differ' $O/id_fix16.log | sed 's/.*: //' | tr '\n' ' ')"
for arm in h f; do
  run $(tree_of $arm) $SMALL 8 $K/lane_thread_census.py --max-concurrency 8 > $O/census16_$arm.log 2>&1
  log "census 16 cpus $arm rc $? $(grep 'total threads' $O/census16_$arm.log)"
done

stop() {
  local name=$1 tree=$2 port=$3
  note $name $tree "colocated, Ctrl-C with 16 videos in flight"
  taskset -c $CPUS python3 $K/stop_under_load.py --tree $tree --model omni_colocated --card 0 --port $port \
    --out $O/$name --serve-extra "$CONTEXT_32K" --videos 16 --after 40 > $O/$name.driver.log 2>&1
  log "$(tail -1 $O/$name.driver.log)"
}
quiet_host
stop st1_m $MAIN 24000
stop st2_h $HEAD 24200
stop st3_f $FIX 24400
stop st4_f $FIX 24600
stop st5_h $HEAD 24800
stop st6_m $MAIN 25000

gates() {
  local name=$1 tree=$2 port=$3
  note $name $tree "CI speech pipeline, text Video-MME c16 x6, talker test x6"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card 0 --port $port --out $O/$name \
    --bench videomme_text --loads 16 --passes clean --bench-repeats 6 \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 6" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
quiet_host
gates g1_m $MAIN 25200
gates g2_f $FIX 25400
gates g3_f $FIX 25600
gates g4_m $MAIN 25800
python3 $K/videomme_answer_diff.py $O main=g1_m/c16_clean,g4_m/c16_clean fix=g2_f/c16_clean,g3_f/c16_clean \
  > $O/gate_text.txt 2>&1
python3 $K/videomme_answer_diff.py $O main=g1_m/talker,g4_m/talker fix=g2_f/talker,g3_f/talker > $O/gate_talker.txt 2>&1
log "gates read: text $(grep -c ' acc ' $O/gate_text.txt) passes, talker $(grep -c ' acc ' $O/gate_talker.txt) passes"

# note (ratish): the CI serves Video-AMME at c16 from two FP8 colocated replicas behind its router, so one replica here
# takes c8, the load each CI replica sees.
amme() {
  local name=$1 tree=$2 port=$3
  note $name $tree "FP8 colocated, Video-AMME c8 x6"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_fp8_colocated --card 0 --port $port --out $O/$name \
    --bench videoamme --loads 8 --passes clean --bench-repeats 6 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
quiet_host
amme a1_m $MAIN 26000
amme a2_f $FIX 26200
amme a3_f $FIX 26400
amme a4_m $MAIN 26600
python3 $K/videomme_answer_diff.py $O main=a1_m/c8_clean,a4_m/c8_clean fix=a2_f/c8_clean,a3_f/c8_clean \
  > $O/gate_amme.txt 2>&1
log "amme read: $(grep -c ' acc ' $O/gate_amme.txt) passes"

export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
video_loop() {
  local name=$1 tree=$2 driver=$3 port=$4 cpus=$5
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
}
mixed() {
  local name=$1 tree=$2 port=$3 server_cpus=$4
  note $name $tree "colocated, speech 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s, server cpus $server_cpus"
  taskset -c $CLIENTS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --server-cpus $server_cpus --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    "--bench-extra=--request-rate=8" --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $CLIENTS &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
  log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)"
}
CLIENTS=$CPUS
quiet_host
mixed e1_m $MAIN 26800 $CPUS
mixed e2_f $FIX 27000 $CPUS
mixed e3_f $FIX 27200 $CPUS
mixed e4_m $MAIN 27400 $CPUS
python3 $K/body_table.py $O c64 "the review fixes against main, 8/s beside video 1/s, colocated" main=e1_m,e4_m \
  fix=e2_f,e3_f >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O e1_m e4_m e2_f e3_f >> $O/queue.log 2>&1
python3 $K/chunk_gap_parts.py --cell c64_events $O/e1_m $O/e4_m $O/e2_f $O/e3_f > $O/chunk_gap_parts_eq.txt 2>&1
log "equal load read"

CLIENTS=64-111,176-223
quiet_host
mixed s1_m $MAIN 27600 $SMALL
mixed s2_f $FIX 27800 $SMALL
python3 $K/body_table.py $O c64 "the review fixes against main, 8/s beside video 1/s, colocated, 16 CPUs" main=s1_m \
  fix=s2_f >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O s1_m s2_f >> $O/queue.log 2>&1
log "small host read"
log "q1127_a done"
