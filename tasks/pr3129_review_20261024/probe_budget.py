"""Run only on the task's Radix CUDA device; record real allocator boundaries."""

import gc
import json
from types import SimpleNamespace

import torch

from sglang_omni.models.fun_cosyvoice3 import stages


def build_pool(**overrides):
    attention = SimpleNamespace(heads=2, inner_dim=8)
    flow = SimpleNamespace(
        decoder=SimpleNamespace(
            estimator=SimpleNamespace(
                transformer_blocks=[SimpleNamespace(attn=attention)] * 2
            )
        )
    )
    options = dict(
        size_gb=None,
        max_blocks=None,
        stage_reserve_bytes=0,
        headroom_bytes=0,
        share_ceiling_bytes=None,
    )
    options.update(overrides)
    return stages.build_prefix_pool(
        flow, torch.device("cuda:0"), torch.bfloat16, **options
    )


def main():
    device = torch.device("cuda:0")
    torch.cuda.init()
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.empty_cache()
    block_bytes = stages.BLOCK_FRAMES * stages.PrefixKVPool.bytes_per_frame(
        layer_num=2, euler_steps=10, head_num=2, head_dim=4, dtype=torch.bfloat16
    )
    pool = build_pool(size_gb=3 * block_bytes / 2**30)
    allocated = sum(
        t.numel() * t.element_size()
        for rows in (pool.keys, pool.values)
        for row in rows
        for t in row
    )
    print(
        json.dumps(
            dict(
                case="explicit_cap",
                cap_bytes=3 * block_bytes,
                actual_tensor_bytes=allocated,
                usable_frames=pool.free_frames,
            )
        ),
        flush=True,
    )
    del pool
    gc.collect()
    torch.cuda.empty_cache()
    cap_bytes = 2 * 2**20
    torch.cuda.set_per_process_memory_fraction(cap_bytes / total, device)
    try:
        pool = build_pool()
    except torch.OutOfMemoryError as error:
        print(
            json.dumps(
                dict(
                    case="hard_cap",
                    cap_bytes=cap_bytes,
                    outcome=type(error).__name__,
                    error=str(error),
                )
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                dict(case="hard_cap", outcome="allocated", frames=pool.free_frames)
            ),
            flush=True,
        )
        del pool
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, device)
    gc.collect()
    torch.cuda.empty_cache()
    previous_stage_temporary = torch.empty(
        256 * 2**20, dtype=torch.uint8, device=device
    )
    del previous_stage_temporary
    baseline = torch.accelerator.memory_reserved(device)
    stage_budget = 128 * 2**20
    share_ceiling = baseline + stage_budget
    pool = build_pool(share_ceiling_bytes=share_ceiling)
    allocated = sum(
        t.numel() * t.element_size()
        for rows in (pool.keys, pool.values)
        for row in rows
        for t in row
    )
    print(
        json.dumps(
            dict(
                case="stale_baseline",
                previous_reserved=baseline,
                stage_budget=stage_budget,
                actual_pool_tensor_bytes=allocated,
                actual_reserved=torch.cuda.memory_reserved(device),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
