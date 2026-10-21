#!/usr/bin/env bash
# 10-21, the 0.5.22 container, cards 0 and 1: #3029's review fixes. Phase base: warms of t_main (upstream main
# 057d97629) and t_h0 (#3029 before the fixes, fd2139ac6); the chunk probe on t_h0 at 1 and 16 rows with the spectrum
# aligned to 8 channels (24, the head) and to 4 (20), the alignment bench; t_h0's seeded decode outputs saved as the
# bitwise reference for the fixed head; the probe on t_main (the channel-first decode on cuDNN 9.24).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261021b/queue_h.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_h_c20261021b.sh base'
D=/data/c20261021b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
CKPT=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512'))")
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_h.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
probe() {
  local tree=$1 card=$2 name=$3
  shift 3
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c $(cpus $card) \
    python3 $IN/probe_hift_chunk_c20261021.py --checkpoint $CKPT "$@" > $D/probe_$name.log 2>&1)
  log "probe $name rc $? $(grep -h gpu_ms_per_chunk $D/probe_$name.log)"
}
tree_log() { log "tree $1 $(git -C /workspace/$1 log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$1 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$1 diff --quiet HEAD -- sglang_omni || echo DIRTY)"; }
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
score() {
  local dir=$1 card=$2 port=$3
  (cd /workspace/t_main && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --transcribe-only --skip-gpu-cleanup --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --host 127.0.0.1 --port $port \
    --lang en --output-dir $dir > $dir/score_wer.log 2>&1)
  log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  (cd /workspace/t_main && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
    --similarity-only --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
  log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
}

if [ "$1" = base ]; then
  log "queue base start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
  tree_log t_main
  tree_log t_h0
  bash $IN/warm_c20261021.sh t_main 0 $(cpus 0) 50000 $D/warm_t_main &
  bash $IN/warm_c20261021.sh t_h0 1 $(cpus 1) 50400 $D/warm_t_h0 &
  wait
  for tree in t_main t_h0; do
    log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
  done
  probe t_h0 0 h0_r1 --rows 1 --save $D/decode_t_h0.pt &
  probe t_h0 1 h0_r1_a4 --rows 1 --channel-alignment 4 &
  wait
  probe t_h0 0 h0_r16 --rows 16 &
  probe t_h0 1 h0_r16_a4 --rows 16 --channel-alignment 4 &
  wait
  probe t_main 0 main_r1 --rows 1 &
  probe t_main 1 main_r16 --rows 16 &
  wait
  log "queue base done"
elif [ "$1" = fix ]; then
  # Phase fix: t_h1, #3029 with the review fixes (tmp/hift-channels-last-h1b); the touched suites on t_main and t_h1;
  # the chunk probe on t_h1 and its seeded outputs against t_h0's, bit for bit; memory after the build on both; a
  # warm of t_h1; p_h1 t_h0 against t_h1 (c1f seeded for byte identity, c16); p_h2 t_main against t_h1 (c1f, c16,
  # the body's table) with its c16 audio scored.
  log "queue fix start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
  cd /workspace/sglang-omni
  git fetch -q origin tmp/hift-channels-last-h1b
  [ -d /workspace/t_h1 ] || git worktree add -q --detach /workspace/t_h1 FETCH_HEAD
  tree_log t_h1
  SUITES="tests/unit_test/fun_cosyvoice3 tests/unit_test/utils/test_channels_last_conv.py
    tests/unit_test/fishaudio_drama3/test_vocoder.py tests/unit_test/minicpm_o/test_dit.py
    tests/unit_test/minicpm_o/test_code2wav.py tests/unit_test/minicpm_o/test_vocoder_batch.py
    tests/unit_test/minicpm_o/test_flow_execution.py tests/unit_test/minicpm_o/test_flow_ragged.py
    tests/unit_test/minicpm_o/test_fixed_packed_graph.py tests/unit_test/qwen3_omni/test_code2wav.py
    tests/unit_test/qwen3_omni/test_code2wav_batching.py tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py
    tests/unit_test/qwen3_omni/test_code2wav_overlap.py tests/unit_test/qwen3_omni/test_code2wav_snake_beta.py
    tests/unit_test/qwen3_omni/test_streaming.py tests/unit_test/qwen3_omni/test_talker_codec_coalesce.py
    tests/unit_test/qwen3_tts/test_incremental_codec.py tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py
    tests/unit_test/qwen3_tts/test_pipeline.py"
  mkdir -p $D/units
  card=0
  for tree in t_main t_h1; do
    (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree:$COSYVOICE taskset -c $(cpus $card) \
      python3 -m pytest -q -p no:cacheprovider $SUITES > $D/units/units_$tree.log 2>&1) &
    card=1
  done
  wait
  for tree in t_main t_h1; do
    log "units $tree $(grep -E '[0-9]+ (passed|failed)' $D/units/units_$tree.log | tail -n 1)"
    grep -E '^(FAILED|ERROR) ' $D/units/units_$tree.log | sort > $D/units/failures_$tree.txt
  done
  log "failures only on t_h1: $(comm -13 $D/units/failures_t_main.txt $D/units/failures_t_h1.txt | tr '\n' ' ' | cut -c1-600)"
  probe t_h1 0 h1_r1 --rows 1 --save $D/decode_t_h1.pt &
  probe t_h1 1 h1_r16 --rows 16 &
  wait
  probe t_h0 0 h0_memory --rows 1 --runs 2 &
  probe t_h1 1 h1_memory --rows 1 --runs 2 &
  wait
  log "memory after build: t_h0 $(grep -ho '"allocated_mib_after_build": [0-9.]*' $D/probe_h0_memory.log) t_h1 $(grep -ho '"allocated_mib_after_build": [0-9.]*' $D/probe_h1_memory.log)"
  (cd /workspace/t_h1 && python3 $IN/probe_hift_chunk_c20261021.py --compare $D/decode_t_h0.pt $D/decode_t_h1.pt > $D/decode_compare.txt 2>&1)
  log "decode t_h0 against t_h1: $(grep -c ' equal ' $D/decode_compare.txt) equal, $(grep -c DIFFERENT $D/decode_compare.txt) different"
  bash $IN/warm_c20261021.sh t_h1 1 $(cpus 1) 50400 $D/warm_t_h1
  log "warm t_h1 $(grep 'bench rc' $D/warm_t_h1/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_t_h1/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_t_h1/serve.log)"
  pair t_h0 t_h1 p_h1 c1f,c16
  identity p_h1 c1f
  pair t_main t_h1 p_h2 c1f,c16
  score $D/p_h2/a/gpu0/c16 0 26000 &
  score $D/p_h2/a/gpu1/c16 1 27000 &
  wait
  score $D/p_h2/b/gpu0/c16 0 26000 &
  score $D/p_h2/b/gpu1/c16 1 27000 &
  wait
  log "queue fix done"
else
  log "unknown phase $1"
fi
