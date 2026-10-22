#!/usr/bin/env bash
# 10-22, cards 0 and 1: the one test of the stack (tasks/cosyvoice-perf/profile_20261022/PLAN.md 7). t_m1b, main at
# the chain's base 1b726ab64, against t_chain (perf/cosyvoice3-items-chain at its latest commit): streaming c1 seeded,
# c16, c64 and non-streaming c1 seeded and c16 over the full SeedTTS EN corpus, both card orders; a main against
# main A/A of the two non-streaming cells (the streaming band is p_aa1's); every second's host load; then WER and SIM
# of every arm's cells from the saved audio, the per clip SIM of the seeded cells, the pair audit and the quality
# summary.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_final.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_final.sh'
D=/data/c20261022b
F=$D/final
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
MODEL=FunAudioLLM/Fun-CosyVoice3-0.5B-2512
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
mkdir -p $F
log() { echo "$* $(date +%T)" >> $F/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
tree() {
  local name=$1 sha=$2
  if [ -d /workspace/$name ]; then
    git -C /workspace/$name checkout -q --detach $sha
  else
    git -C /workspace/sglang-omni worktree add -q --detach /workspace/$name $sha
  fi
  log "tree $name $(git -C /workspace/$name log --oneline -1 | cut -c1-60) sglang_omni $(git -C /workspace/$name rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$name diff --quiet HEAD -- sglang_omni || echo DIRTY)"
}
git -C /workspace/sglang-omni fetch -q upstream main
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-items-chain
chain=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
tree t_m1b 1b726ab64
tree t_chain $chain
log "queue final start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
# note (ratish): Inductor's cache keys a custom op by name, not schema, so code compiled while L7a gave
# fun_cosyvoice3_packed_fa3 a ninth argument came back after its revert (10-22); the units and the chain's
# server compile into fresh caches.
rm -rf $F/inductor_units $D/inductor_t_chain
(cd /workspace/t_chain && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$F/inductor_units \
  PYTHONPATH=/workspace/t_chain:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 tests/unit_test/test_platforms.py -q -p no:cacheprovider -rfEs \
  > $F/units.log 2>&1)
units_rc=$?
log "units rc $units_rc $(grep -E 'passed|failed' $F/units.log | tail -n 1)"
[ $units_rc = 0 ] || { log "units failed, stop"; exit 1; }
bash $IN/warm_c20261021.sh t_m1b 0 $(cpus 0) 29400 $D/warm_t_m1b &
bash $IN/warm_c20261021.sh t_chain 1 $(cpus 1) 29600 $D/warm_t_chain_final &
wait
for w in warm_t_m1b warm_t_chain_final; do
  log "warm $w $(grep 'bench rc' $D/$w/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/$w/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/$w/serve.log)"
done
pair() {
  local first=$1 second=$2 name=$3 cells=$4
  python3 $IN/host_load_c20261022.py $F/${name}_load.jsonl &
  local sampler=$!
  (cd /workspace/sglang-omni && PAIR_CELLS=$cells python3 $IN/serve_pair_c20261021.py /workspace/$first \
    /workspace/$second $F/$name > $F/$name.log 2>&1)
  log "pair $name rc $? token cap hits $(grep -E '\| at the token cap \|' $F/$name.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
  kill $sampler
  python3 $IN/host_load_c20261022.py --summary $F/${name}_load.jsonl $F/$name $cells > $F/${name}_load.txt 2>&1
}
pair t_m1b t_chain p_f1 c1f,c16,c64,b1f,b16
pair t_m1b t_m1b p_f0 b1f,b16
score() {
  local dir=$1 card=$2 port=$3
  if [ ! -s $dir/wer_results.json ]; then
    (cd /workspace/t_m1b && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --transcribe-only --skip-gpu-cleanup --model $MODEL --host 127.0.0.1 --port $port --lang en --output-dir $dir \
      > $dir/score_wer.log 2>&1)
    log "wer $dir rc $? $(grep 'WER mean' $dir/score_wer.log | tr -s ' ')"
  fi
  if [ ! -s $dir/similarity_results.json ]; then
    (cd /workspace/t_m1b && CUDA_VISIBLE_DEVICES=$card taskset -c $(cpus $card) python3 -u -m benchmarks.eval.benchmark_tts_seedtts \
      --similarity-only --model $MODEL --lang en --output-dir $dir > $dir/score_sim.log 2>&1)
    log "sim $dir rc $? $(grep 'speaker similarity' $dir/score_sim.log | cut -c1-40)"
  fi
}
round=0
for cell in c1f c16 c64 b1f b16; do
  for order in a b; do
    round=$((round + 1))
    score $F/p_f1/$order/gpu0/$cell 0 $((26000 + 600 * round)) &
    score $F/p_f1/$order/gpu1/$cell 1 $((26300 + 600 * round)) &
    wait
  done
done
for cell in c1f b1f; do
  python3 $IN/paired_sim_c20261021.py $F/p_f1/a/gpu0/$cell $F/p_f1/a/gpu1/$cell > $F/paired_sim_${cell}_a.txt 2>&1
  python3 $IN/paired_sim_c20261021.py $F/p_f1/b/gpu1/$cell $F/p_f1/b/gpu0/$cell > $F/paired_sim_${cell}_b.txt 2>&1
  log "paired sim $cell a $(tr '\n' ' ' < $F/paired_sim_${cell}_a.txt | cut -c1-260)"
  log "paired sim $cell b $(tr '\n' ' ' < $F/paired_sim_${cell}_b.txt | cut -c1-260)"
done
python3 $HERE/pair_audit_c20261022.py $F /workspace/sglang-omni > $F/pair_audit.txt 2>&1
python3 $HERE/quality_summary_c20261022.py $F p_f1:c1f,c16,c64,b1f,b16 > $F/quality.txt 2>&1
log "audit and quality rc $?"
log "queue final done"
