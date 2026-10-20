#!/usr/bin/env bash
# E5: which code makes the memory that grows while serving, beyond SeedTTS English. Prerequisites 1 and 2
# with item 3 v3 (t_p123v3) at 16 vocoder rows on both cards, the cells of e5_cells_c20261011.txt: SeedTTS
# English and Chinese, references of 28 s, of 1.5 s, at 48 kHz and at 8 kHz, instructions, buffered c32, long
# form, and the true worst step (16 identical requests at the text limit, new tokens up to 3,000 over about
# 1,000 prompt tokens, so each row nears the AR's 4,096 context). Card 0 records the allocator's history with
# Python stacks from healthy and dumps a snapshot after each cell. Card 1 runs with CUDA_MODULE_LOADING=EAGER
# and no history: if the memory outside torch stops growing there, lazy kernel loading is what grows it.
# usage (in the container): bash queue_e5_c20261011.sh
R=/data/c20261011
IN=$R/in
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
mkdir -p $R/cores $R/e5
if [ ! -f $R/mixed/sr8k/meta.lst ]; then
  (cd /workspace/sglang-omni && PYTHONPATH=/workspace/sglang-omni python3 $IN/make_mixed_meta_c20261011.py \
    --out $R/mixed) > $R/e5/lists.log 2>&1
fi

arm() {
  local card=$1 port=$2 cpus=$3 out=$4 cache=$5 env_extra=$6
  mkdir -p "$out"
  EXTRA_ENV="PYTHONPATH=/workspace/t_p123v3:$IN/e5_probe \
E5_KIT_SITECUSTOMIZE=$KIT/prefix_count/sitecustomize.py E5_OUT=$out/e5 E5_PORT=$port \
E5_PROGRESS=$out/progress.txt OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t $env_extra" \
    bash $IN/e5_boot_c20261011.sh /workspace/t_p123v3 "$out" "$card" "$port" "$cpus" "$cache" \
    $IN/e5_cells_c20261011.txt
}

echo "e5 start $(date +%T)" >> $R/queue_e5.txt
arm 0 36000 0-15,64-79 $R/e5/hist $R/inductor_t_p123v3 "E5_HISTORY=1" &
arm 1 36200 16-31,80-95 $R/e5/eager $R/inductor_t_p123v3_e5b "CUDA_MODULE_LOADING=EAGER" &
wait
echo "e5 done $(date +%T)" >> $R/queue_e5.txt
