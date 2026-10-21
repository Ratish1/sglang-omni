#!/usr/bin/env bash
# 10-21, cards 0 and 1: what is left of queues u and v, cut to the runs that decide, each PR judged by its own head
# against ours, at c1 and c16 only. Queue u's bash was stopped while its pair p_u1 (main against #2260 as is) ran on;
# this queue stops that pair once its second order has b16, writes its table for b1f and b16, and scores its b16 audio
# (the PR's own claim). #2260: p_x1 the PR with main merged (t_b0) against our branch (t_b1) at b1f, seeded for byte
# identity, and b16; p_u2 the one call arm at b16 (at c1 a batch has one request, nothing to group); the profiled b16
# pair main against t_b1. #2944: the PR with main merged (t_p44, tmp/pr-2944-main 1b0e74b65) against ours (t_s) at
# b1f for byte identity; the startup check t_b1 against t_s (one boot per card order), with streaming c1f and c16 only
# if a pool's size moved. Cut as deciding nothing: b64 and long form (the vocoder batches 16 requests whatever the
# concurrency, and the route and the grouping act the same on long rows), #2260 on #3019 (#2260 lands on main), a
# second boot per order, c1f identity (b1f reads the RoPE angles every packed path takes).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261021/queue_x.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_x_c20261021.sh'
D=/data/c20261021
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner tests/unit_test/scheduling
  tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py"
log() { echo "$* $(date +%T)" >> $D/queue_x.txt; }
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
identity() {
  local name=$1 cell=$2
  python3 $IN/audio_identity_c20261020.py $D/$name $cell > $D/identity_${name}_$cell.txt 2>&1
  log "identity $name $cell $(tr '\n' ' ' < $D/identity_${name}_$cell.txt | cut -c1-300)"
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
live() { ps -o pid=,stat= -g "$1" | awk '$2 !~ /^Z/ {print $1}'; }

U=$(cat $D/queue_u.pgid)
until [ -s $D/p_u1/b/gpu0/b16/speed_results.json ] && [ -s $D/p_u1/b/gpu1/b16/speed_results.json ]; do
  [ -z "$(live $U)" ] && break
  sleep 20
done
# note (ratish): SIGINT reaches the pair driver's finally, which stops both servers by their process groups.
pkill -INT -g $U -f serve_pair_c20261021.py
for i in $(seq 1 30); do [ -z "$(live $U)" ] && break; sleep 5; done
[ -n "$(live $U)" ] && kill -TERM $(live $U)
(cd $IN && PAIR_CELLS=b1f,b16 python3 -c "import sys; from pathlib import Path; import serve_pair_c20261021 as pair; pair.report(Path('$D/p_u1'), ('t_m22', 't_b0'))" > $D/p_u1.log 2>&1)
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD); p_u1 stopped after b16, table rows $(grep -c '^| [ab] |' $D/p_u1.log)"
for dir in a/gpu0 a/gpu1 b/gpu0 b/gpu1; do score $D/p_u1/$dir/b16; done
pair t_b0 t_b1 p_x1 b1f,b16
identity p_x1 b1f
pair t_b1 t_b1c p_u2 b16
profiled t_m22 t_b1 prof_u1 b16:16:320::::b

cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-skip-native-startup tmp/pr-2944-main
for spec in t_p44:1b0e74b654cfe85a40a4377a29c8d3eff15a769f t_s:0a172c2d8; do
  tree=${spec%%:*} commit=${spec#*:}
  [ -d /workspace/$tree ] || git worktree add -q --detach /workspace/$tree $commit
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
bash $IN/warm_c20261021.sh t_p44 0 0-15,64-79 50000 $D/warm_t_p44 &
bash $IN/warm_c20261021.sh t_s 1 16-31,80-95 50400 $D/warm_t_s &
wait
card=0
for tree in t_p44 t_s; do
  cpus=$([ $card = 0 ] && echo 0-15,64-79 || echo 16-31,80-95)
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c $cpus \
    python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_$tree.log 2>&1) &
  card=1
done
wait
for tree in t_p44 t_s; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
  log "units $tree $(grep -E 'passed|failed' $D/units/units_$tree.log | tail -n 1)"
done
pair t_p44 t_s p_x2 b1f
identity p_x2 b1f
(cd /workspace/sglang-omni && python3 $IN/startup_c20261021.py /workspace/t_b1 /workspace/t_s $D/startup_v 1 \
  > $D/startup_v.log 2>&1)
log "startup rc $? $(grep -E 'mean' $D/startup_v.log | tr '\n' ' ')"
pools_moved=$(python3 -c "
import json
rows = json.load(open('$D/startup_v/startup.json'))
print(int(len({tuple(row['pools']) for row in rows}) != 1))")
log "pools moved $pools_moved"
if [ "$pools_moved" = 1 ]; then
  pair t_b1 t_s p_v1 c1f,c16
else
  log "no pool moved, streaming untouched"
fi
log "queue done"
