# SPDX-License-Identifier: Apache-2.0
"""Shared test doubles."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace


class FakeExecutionBridge:
    """SGLangExecutionBridge double for scheduler-owned ModelRunner tests."""

    def __init__(self, device: object | None = None) -> None:
        import torch

        self.published: list[tuple[object, object]] = []
        self.isolate_sampling_calls: list[bool] = []
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.device_module = torch.get_device_module(self.device)

    @contextlib.contextmanager
    def forward_context(self, batch: object, *, isolate_sampling: bool = False):
        del batch
        self.isolate_sampling_calls.append(isolate_sampling)
        yield

    def publish_next_tokens(self, batch: object, next_token_ids: object) -> None:
        self.published.append((batch, next_token_ids))

    def record_completion(self):
        return self.device_module.Event()


class FakeServerArgs(SimpleNamespace):
    """ServerArgs double exposing the 0.5.16 override() mutation entry point."""

    def override(self, source: str, **fields: object) -> None:
        del source
        for name, value in fields.items():
            setattr(self, name, value)


def real_radix_pools(size: int = 64):
    """CPU-resident upstream KV allocator, request pool and radix cache."""
    import torch
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixCache

    kv = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=torch.float16,
        head_num=1,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=size, dtype=torch.float16, device="cpu", kvcache=kv, need_sort=False
    )
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    return allocator, req_to_token_pool, cache
