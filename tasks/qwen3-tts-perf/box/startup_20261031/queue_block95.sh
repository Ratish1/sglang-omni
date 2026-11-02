#!/usr/bin/env bash
# Block 95 of 11-02: is compiling the wide windows worth a larger ring. The whole request head against the same head with
# every window width in place and compiled. codec_kernels.py on both at the same time, each in its own cache: build, replay
# and bits per graph (streaming's cold and follow-up graphs must stay bit-equal, or say what moved); then non-streaming
# concurrency 64 on the wide tree alone, read against block 92's 75.3 req/s. Nothing scored until this says it pays.
# usage: D=... T=<kit tree> WHOLE=<tree> WIDE=<tree> bash queue_block95.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
WHOLE=${WHOLE:?the base tree}
WIDE=${WIDE:?the tree under test}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block95
L=$W/queue.log
mkdir -p $W
for arm in WHOLE WIDE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block95 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
kernels() {
  local label=$1 tree=$2 card=$3
  (cd $tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_$label/inductor TRITON_CACHE_DIR=$W/cache_$label/triton \
    timeout 2400 python3 $K/startup_20261031/codec_kernels.py --all-widths --out $W/$label > $W/$label.log 2>&1)
  echo "kernels $label exit $? $(date +%T)" >> $L
}
kernels whole $WHOLE 0 &
kernels wide $WIDE 1 &
wait
python3 - $W >> $L 2>&1 <<'EOF'
import json
import sys

import torch

root = sys.argv[1]
a, b = json.load(open(f"{root}/whole.json")), json.load(open(f"{root}/wide.json"))
print(f"build {a['build_s']} / {b['build_s']} s; programs {a['programs']} / {b['programs']}")
wave_a, wave_b = torch.load(f"{root}/whole.pt"), torch.load(f"{root}/wide.pt")
for family in ("cold", "warm0", "window"):
    keys = [key for key in a["keys"] if key.startswith(family)]
    same = sum(torch.equal(wave_a[key], wave_b[key]) for key in keys)
    print(f"{family}: {len(keys)} keys, bit-equal {same}, replay summed {sum(a['keys'][k]['ms'] for k in keys):.2f} / {sum(b['keys'][k]['ms'] for k in keys):.2f} ms")
for width in (16, 32, 64):
    for batch in (1, 2, 4, 8):
        key = f"window:{width}x{batch}"
        distance = float((wave_a[key] - wave_b[key]).norm() / wave_a[key].norm().clamp_min(1e-9))
        print(f"{key}: {a['keys'][key]['ms']:.2f} / {b['keys'][key]['ms']:.2f} ms ({100 * (b['keys'][key]['ms'] / a['keys'][key]['ms'] - 1):+.1f} %), launches {len(a['keys'][key]['kernels'])} / {len(b['keys'][key]['kernels'])}, relative L2 {distance:.4f}")
EOF
(cd $K/inline_20261026 && ARM_ROOT=$W/arms TORCHINDUCTOR_CACHE_DIR=$W/cache_wide/inductor TRITON_CACHE_DIR=$W/cache_wide/triton timeout 1800 python3 serve_arm.py --label wide_c64 --tree $WIDE --card 0 --port 20000 --concurrency 64 --no-stream > $W/wide_c64.out 2>&1)
echo "wide_c64 exit $? $(date +%T) startup $(cat $W/arms/wide_c64/startup.txt 2>/dev/null) s; $(grep -hE 'Throughput \(req/s\)|RTF mean|Latency p95' $W/arms/wide_c64/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-120); $(grep -m1 -oE '[0-9.]+ MiB per stream, [0-9.]+ MiB total' $W/arms/wide_c64/serve.log); $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/wide_c64/serve.log); tracebacks $(grep -c Traceback $W/arms/wide_c64/serve.log)" >> $L
echo "queue block95 done $(date +%T)" >> $L
