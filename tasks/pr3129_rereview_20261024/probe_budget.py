"""Exercise real prefix allocations and preserve the caller's allocator policy."""

import gc
import json
from types import SimpleNamespace

import torch

from sglang_omni.models.fun_cosyvoice3 import stages


def main():
    device = torch.device("cuda", 0)
    attention = SimpleNamespace(heads=16, inner_dim=1024)
    flow = SimpleNamespace(
        decoder=SimpleNamespace(
            estimator=SimpleNamespace(
                transformer_blocks=[SimpleNamespace(attn=attention)] * 22,
                input_embed=SimpleNamespace(proj=SimpleNamespace(out_features=1024)),
            )
        )
    )
    gc.collect()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(device).total_memory
    prior_fraction = torch.cuda.get_per_process_memory_fraction(device)
    baseline = torch.cuda.memory_reserved(device)
    pool = step = None
    try:
        torch.cuda.set_per_process_memory_fraction(
            (baseline + 8 * 2**30) / total, device
        )
        pool = stages.build_prefix_pool(
            flow,
            device,
            torch.bfloat16,
            size_gb=None,
            max_blocks=None,
            stage_reserve_bytes=6 * 2**30,
            headroom_bytes=0,
            share_ceiling_bytes=None,
        )
        assert pool is not None
        mapped_bytes = torch.cuda.memory_reserved(device) - baseline
        step = torch.empty(6 * 2**30, dtype=torch.uint8, device=device)
        print(
            json.dumps(
                dict(
                    case="reserve",
                    outcome="passed",
                    frames=pool.free_frames,
                    pool_mapped_bytes=mapped_bytes,
                    peak_reserved=torch.cuda.max_memory_reserved(device),
                )
            ),
            flush=True,
        )
    finally:
        step = pool = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(prior_fraction, device)


if __name__ == "__main__":
    main()
