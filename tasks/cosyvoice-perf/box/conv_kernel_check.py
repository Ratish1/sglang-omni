# Fused positional conv kernel facts for the item 2 review: binaries per channel shape across
# frame counts, compiled shared memory, registers and spills per group width, and per conv
# time at serving shapes. Run with PYTHONPATH set to the tree under test.
import sys

import torch
import triton.testing

from sglang_omni.models.fun_cosyvoice3 import causal_conv

print("module", causal_conv.__file__)
device = torch.device("cuda")
kernel = causal_conv.group_conv_mish_kernel
widths = [int(arg) for arg in sys.argv[1:]] or [16, 32, 64, 128]
for group_channels in widths:
    channels = 16 * group_channels
    conv = torch.nn.Conv1d(channels, channels, 31, groups=16).to(device, torch.bfloat16)
    weight = causal_conv.pack_group_conv_weight(conv)
    try:
        for frames in (31, 32, 33, 47, 48, 95, 1000, 1001, 1024):
            x = torch.randn(1, frames, channels, device=device, dtype=torch.bfloat16)
            causal_conv.group_conv_mish(x, weight, conv.bias)
        torch.cuda.synchronize()
        print(f"width {group_channels}: launched 9 frame counts")
    except Exception as error:
        print(f"width {group_channels}: failed {type(error).__name__}: {str(error)[:300]}")

for cache_device, entry in kernel.device_caches.items():
    binaries = entry[0]
    print("device", cache_device, "binaries", len(binaries))
    for key, binary in binaries.items():
        metadata = binary.metadata
        print(
            f"  shared {metadata.shared} B, regs {binary.n_regs}, spills {binary.n_spills}, "
            f"warps {metadata.num_warps}, stages {metadata.num_stages}, key {str(key)[:160]}"
        )

channels = 1024
conv = torch.nn.Conv1d(channels, channels, 31, groups=16).to(device, torch.bfloat16)
weight = causal_conv.pack_group_conv_weight(conv)
for batch, frames in ((1, 1690), (1, 4126), (1, 9216), (16, 1030)):
    x = torch.randn(batch, frames, channels, device=device, dtype=torch.bfloat16)
    milliseconds = triton.testing.do_bench(
        lambda: causal_conv.group_conv_mish(x, weight, conv.bias), warmup=50, rep=300
    )
    print(f"time batch {batch} in frames {frames}: {milliseconds * 1000:.1f} us")
