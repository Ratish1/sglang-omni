#!/usr/bin/env bash
# E2: what grows after item 3's pool is sized, per stream and outside torch, on prerequisites 1 and 2
# with item 3 v3 (t_p123v3). Card 0 at the default 16 vocoder rows: stream c16 and c64, buffered c32,
# long form c16 (48 x 10 texts), the length stress c16 (32 x 40 texts) and the admissible worst step
# (16 identical requests at the text limit, seeded, finals in one step). Card 1 at 32 rows: the 32
# request worst step and stream c64. Each server carries the per stream allocator probe (e2_probe) in
# front of the kit's count probe with its memory fields, GPU core dumps armed.
# usage (in the container): bash queue_e2_c20261011.sh
R=/data/c20261011
IN=$R/in
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
mkdir -p $R/cores $R/e2
cd /workspace/sglang-omni
PYTHONPATH=/workspace/sglang-omni python3 $KIT/make_longform_meta.py --out $R/stress \
  --entries 32 --texts-per-entry 40 > $R/e2/lists.log 2>&1
PYTHONPATH=/workspace/sglang-omni python3 $IN/make_worst_meta_c20261011.py --out $R/worst \
  --requests 16 >> $R/e2/lists.log 2>&1
PYTHONPATH=/workspace/sglang-omni python3 $IN/make_worst_meta_c20261011.py --out $R/worst32 \
  --requests 32 >> $R/e2/lists.log 2>&1

arm() {
  local card=$1 port=$2 cpus=$3 out=$4 cache=$5 cells=$6 serve_args=${7:-}
  mkdir -p "$out"
  EXTRA_ENV="PYTHONPATH=/workspace/t_p123v3:$IN/e2_probe \
E2_KIT_SITECUSTOMIZE=$KIT/prefix_count/sitecustomize.py E2_STREAM_PROBE=1 E2_STREAM_PROBE_OUT=$out/streams \
OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t" \
    CELLS="$cells" SERVE_ARGS="$serve_args" \
    bash $KIT/ab_boot.sh /workspace/t_p123v3 "$out" "$card" "$port" "$cpus" "$cache"
}

echo "e2 start $(date +%T)" >> $R/queue_e2.txt
arm 0 35000 0-15,64-79 $R/e2/r16 $R/inductor_t_p123v3 \
  "s16:16:::1 s64:64:::1 b32:32:::0 L16:16:::1:$R/longform/meta.lst X16:16:::1:$R/stress/meta.lst W16:16:16:1234:1:$R/worst/meta.lst" &
arm 1 35200 16-31,80-95 $R/e2/r32 $R/inductor_t_p123v3_r32 \
  "W32:32:32:1234:1:$R/worst32/meta.lst s64:64:::1" "--vocoder.factory.max_batch_size 32" &
wait
echo "e2 done $(date +%T)" >> $R/queue_e2.txt
