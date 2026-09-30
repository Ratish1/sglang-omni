#!/usr/bin/env bash
# Bring a fresh container to the Qwen3-Omni census and start it (run inside the container):
# the main and tools trees, main's pinned dependencies, the checkpoint and the corpora, the
# known answers on the fourth card, then the census matrices on cards 0 to 2, three servers
# at most: card 0 the default profile, card 1 both prefill graphs, card 2 the clean baselines
# (K10) and the capture label boots (attribution). Everything lands under OUT.
# usage: census_bringup.sh <out>
set -u
OUT=$1
mkdir -p "$OUT"
cd /workspace/sglang-omni
git fetch -q origin analysis/qwen3-omni-perf-20260923
git show FETCH_HEAD:tasks/qwen3_omni_perf_20260923/scripts/setup_trees.sh > /tmp/setup_trees.sh
bash /tmp/setup_trees.sh main=upstream/main tools=origin/analysis/qwen3-omni-perf-20260923 > "$OUT/trees.txt" 2>&1
MAIN=/workspace/sglang-omni/.tmp/wt/main
S=/workspace/sglang-omni/.tmp/wt/tools/tasks/qwen3_omni_perf_20260923/scripts
(cd "$MAIN" && pip install -e . > "$OUT/pip.log" 2>&1)
python3 -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)" >> "$OUT/trees.txt"
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct > "$OUT/model.log" 2>&1
(cd "$MAIN" && PYTHONPATH=$MAIN python3 "$S/prefetch.py" seedtts mmsu mmmu > "$OUT/prefetch.log" 2>&1)
echo "setup done $(date +%T)" >> "$OUT/bringup.log"

mkdir -p "$OUT/ka"
(cd "$OUT/ka" && CUDA_VISIBLE_DEVICES=3 nsys profile -o ka --force-overwrite=true \
  --trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=node --gpuctxsw=true --sample=none --cpuctxsw=none \
  python3 "$S/omni_known_answer.py" run > run.log 2>&1 \
  && nsys export --type sqlite --force-overwrite=true -o ka.sqlite ka.nsys-rep > export.log 2>&1 \
  && python3 "$S/omni_known_answer.py" analyze ka.sqlite > analyze.txt 2>&1)
echo "known answers done $(date +%T)" >> "$OUT/bringup.log"

cells() {
  local config=$1
  echo "$config:seedtts_en:16:probe:128 $config:seedtts_en_nostream:16:probe:128 $config:mmsu_talker:16:probe:128 $config:mmmu_talker:16:probe:48 $config:seedtts_en:1:probe:48"
}
bash "$S/census_matrix.sh" "$MAIN" "$OUT/card0" 0 8000 "$(cells default)" > "$OUT/card0.log" 2>&1 &
bash "$S/census_matrix.sh" "$MAIN" "$OUT/card1" 1 9000 "$(cells both)" > "$OUT/card1.log" 2>&1 &
bash "$S/census_matrix.sh" "$MAIN" "$OUT/card2" 2 10000 \
  "default:seedtts_en:16:clean:128 both:seedtts_en:16:clean:128 default:seedtts_en:16:lines:128 both:seedtts_en:16:lines:128" \
  > "$OUT/card2.log" 2>&1 &
echo "matrices started $(date +%T)" >> "$OUT/bringup.log"
wait
echo "all done $(date +%T)" >> "$OUT/bringup.log"
