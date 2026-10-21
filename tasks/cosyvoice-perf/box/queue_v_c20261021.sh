#!/usr/bin/env bash
# 10-21, cards 0 and 1, after queue u. #2944's half on our #2260 branch (PR2260_REVIEW.md 5, 6): t_s
# (perf/cosyvoice3-skip-native-startup 0a172c2d8) skips the padded DiT's compile and graphs on the packed route.
# Its warm and touched suites, the startup check against t_b1 (two boots each card order), and the pair p_v1 t_b1
# against t_s: b1f and c1f seeded for byte identity (the route and its kernels are the same; only startup differs),
# c16 and c64 because freed memory could reach the KV pool.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261021/queue_v.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_v_c20261021.sh'
D=/data/c20261021
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/model_runner tests/unit_test/scheduling
  tests/unit_test/qwen3_omni/test_talker_token_readback.py tests/unit_test/qwen3_tts/test_decode_step_staging.py
  tests/unit_test/moss_tts_local/test_state_pool.py"
log() { echo "$* $(date +%T)" >> $D/queue_v.txt; }
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $D/$name > $D/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $D/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
}

while pgrep -g "$(cat $D/queue_u.pgid)" > /dev/null; do sleep 60; done
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
cd /workspace/sglang-omni
git fetch -q origin perf/cosyvoice3-skip-native-startup
[ -d /workspace/t_s ] || git worktree add -q --detach /workspace/t_s 0a172c2d8
for tree in t_b1 t_s; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done

bash $IN/warm_c20261021.sh t_s 0 0-15,64-79 50000 $D/warm_t_s &
(cd /workspace/t_s && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_s:$COSYVOICE taskset -c 16-31,80-95 \
  python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_t_s.log 2>&1) &
wait
log "warm t_s $(grep 'bench rc' $D/warm_t_s/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_t_s/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_t_s/serve.log)"
log "units t_s $(grep -E 'passed|failed' $D/units/units_t_s.log | tail -n 1)"

(cd /workspace/sglang-omni && python3 $IN/startup_c20261021.py /workspace/t_b1 /workspace/t_s $D/startup_v 2 \
  > $D/startup_v.log 2>&1)
log "startup rc $? $(grep -E 'mean' $D/startup_v.log | tr '\n' ' ')"

pair t_b1 t_s p_v1 b1f,c1f,c16,c64
for cell in b1f c1f; do
  python3 $IN/audio_identity_c20261020.py $D/p_v1 $cell > $D/identity_v1_$cell.txt 2>&1
  log "identity $cell $(tr '\n' ' ' < $D/identity_v1_$cell.txt | cut -c1-300)"
done
log "queue done"
