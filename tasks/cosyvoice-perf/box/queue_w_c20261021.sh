#!/usr/bin/env bash
# 10-21, cards 0 and 1: what is left of queues u and v, cut to the runs that decide. Queue u's bash was stopped while
# its pair p_u1 (main against #2260 as is) ran on; this queue waits for that pair, scores its b16 audio, then runs
# p_u2 (t_b1 against the one call arm at b16 and long form), the profiled b16 pair main against t_b1, #2944's warm,
# units, startup (one boot per card order) and b1f byte identity, its streaming cells only if a pool's size moved,
# and last the profiled c64 pair of L2. Dropped as deciding nothing: b64 (the vocoder batches 16 requests whatever the
# concurrency), #2260 on #3019 (#2260 lands on main), a second boot per order, c1f identity (b1f already reads the RoPE
# angles every packed path takes).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261021/queue_w.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_w_c20261021.sh'
D=/data/c20261021
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner tests/unit_test/scheduling
  tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py"
log() { echo "$* $(date +%T)" >> $D/queue_w.txt; }
score() {
  local dir=$1
  (cd /workspace/t_m22 && CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port 26000 --lang en \
    --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_m22 && CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}
profiled() {
  local first=$1 second=$2 name=$3 cells=$4
  PEER=$D/$name/$second CELLS=$cells PYSPY=0 bash $IN/cosy_profile_run_c20261021.sh $first 0 0-15,64-79 48600 \
    $D/$name/$first &
  PEER=$D/$name/$first CELLS=$cells PYSPY=0 bash $IN/cosy_profile_run_c20261021.sh $second 1 16-31,80-95 49000 \
    $D/$name/$second &
  wait
  log "profiled $name $(tail -n 1 $D/$name/$first/progress.txt) ; $(tail -n 1 $D/$name/$second/progress.txt)"
}

# note (ratish): the container's PID 1 reaps nothing, so a finished queue stays as zombies; wait on live members.
while ps -o stat= -g "$(cat $D/queue_u.pgid)" | grep -qv '^Z'; do sleep 60; done
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD); p_u1 table rows $(grep -c '^| [ab] |' $D/p_u1.log)"
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do score $D/p_u1/$dir/b16; done
pair t_b1 t_b1c p_u2 b16,Lb16
profiled t_m22 t_b1 prof_u1 b16:16:320::::b

cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-skip-native-startup
[ -d /workspace/t_s ] || git worktree add -q --detach /workspace/t_s 0a172c2d8
log "tree t_s $(git -C /workspace/t_s log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/t_s rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_s diff --quiet HEAD -- sglang_omni || echo DIRTY)"
bash $IN/warm_c20261021.sh t_s 0 0-15,64-79 50000 $D/warm_t_s &
(cd /workspace/t_s && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_s:$COSYVOICE taskset -c 16-31,80-95 \
  python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_t_s.log 2>&1) &
wait
log "warm t_s $(grep 'bench rc' $D/warm_t_s/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_t_s/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_t_s/serve.log)"
log "units t_s $(grep -E 'passed|failed' $D/units/units_t_s.log | tail -n 1)"
(cd /workspace/sglang-omni && python3 $IN/startup_c20261021.py /workspace/t_b1 /workspace/t_s $D/startup_v 1 \
  > $D/startup_v.log 2>&1)
log "startup rc $? $(grep -E 'mean' $D/startup_v.log | tr '\n' ' ')"
pools_moved=$(python3 -c "
import json
rows = json.load(open('$D/startup_v/startup.json'))
print(int(len({tuple(row['pools']) for row in rows}) != 1))")
log "pools moved $pools_moved"
if [ "$pools_moved" = 0 ]; then
  pair t_b1 t_s p_v1 b1f
else
  pair t_b1 t_s p_v1 b1f,c16,c64
fi
python3 $IN/audio_identity_c20261020.py $D/p_v1 b1f > $D/identity_v1_b1f.txt 2>&1
log "identity b1f $(tr '\n' ' ' < $D/identity_v1_b1f.txt | cut -c1-300)"
profiled t_pr t_l2 prof_u2 c64:64:640
log "queue done"
