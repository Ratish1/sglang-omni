#!/usr/bin/env bash
# 11-19, node-0001, one H200 (card 0, NUMA 0, CPUs 0-55,112-167 of 224): #3452 with the video CPU share sized in the
# stage process (B eb449662d) against the PR head (cea7dde64) and main (35890f10c). Every server launched without
# taskset, as the CI launches it, so the launcher sees 224 CPUs and each stage process its GPU-local 112. Units on B;
# the one-CPU, two-video probe on the head and B; the CI's speech pipeline on one card: text Video-MME c1 and c16 and
# the CI's talker Video-MME test at c16 (m h b b h m); SeedTTS c16 beside 4 looping videos on that server
# (m h b b h m); the colocated config's SeedTTS c16 beside 4 videos (m h b b h m).
set -u
O=/data/v1119
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main PRHEAD=/workspace/vhead B=/workspace/vb
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
tree_of() { case $1 in m) echo $MAIN ;; h) echo $PRHEAD ;; b) echo $B ;; *) exit 2 ;; esac; }
log "q1119_stageshare start"

if [ ! -s $O/share_probe.log ]; then
  (cd $B && python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni tests/unit_test/preprocessing \
    tests/unit_test/utils tests/unit_test/minicpm_o tests/unit_test/ming_omni 2>&1 | tail -15) > $O/b_units.log
  log "units $(grep -E 'passed|failed' $O/b_units.log | tail -1)"
  for arm in h b; do
    (cd $(tree_of $arm) && PYTHONPATH=$(tree_of $arm) python3 $K/video_share_probe.py 2>&1 | tail -3) >> $O/share_probe.log
  done
  log "share probe done"
else
  :
fi

ci_text_and_talker() {
  local name=$1 tree=$2 port=28400
  note $name $tree "omni_ci_onecard text video c1 c16, talker c16"
  python3 $K/baseline.py --tree $tree --model omni_ci_onecard --port $port --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 \
    --after "python3 $K/stage_cpus.py \$(cat $O/$name/server.pgid) > $O/$name/stage_cpus.txt; python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 3" \
    > $O/$name.driver.log 2>&1
  local rc=$?
  log "boot $name rc $rc"
  return $rc
}
first=1
for arm in m h b b h m; do
  quiet_host
  index=$([ -d $O/oc_${arm}1 ] && echo 2 || echo 1)
  ci_text_and_talker oc_${arm}$index $(tree_of $arm)
  rc=$?
  if [ $first = 1 ] && [ $rc != 0 ]; then log "q1119_stageshare stopped: first boot failed"; exit 1; else :; fi
  first=0
done
log "one-card ci text and talker done"

mixed() {
  local name=$1 tree=$2 model=$3 extra=$4 port=$5
  note $name $tree "$model seedtts c16 beside text video"
  python3 $K/baseline.py --tree $tree --model $model --port $port --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree python3 $K/videomme_speed.py --host 127.0.0.1 --port $port --repo-id zhaochenyang20/Video_MME_ci \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 4 \
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
for arm in m h b b h m; do
  quiet_host
  index=$([ -d $O/sv_${arm}1 ] && echo 2 || echo 1)
  mixed sv_${arm}$index $(tree_of $arm) omni_ci_onecard "" 28500
done
log "one-card ci speech beside video done"

for arm in m h b b h m; do
  quiet_host
  index=$([ -d $O/cv_${arm}1 ] && echo 2 || echo 1)
  mixed cv_${arm}$index $(tree_of $arm) omni_colocated "$CONTEXT_32K" 28600
done
log "q1119_stageshare done"
