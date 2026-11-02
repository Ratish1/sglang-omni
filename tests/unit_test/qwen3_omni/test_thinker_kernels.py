# SPDX-License-Identifier: Apache-2.0
"""The thinker's decode norm, rotation and store kernel against sglang's three kernels."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.attention.rope import FusedSetKVBufferArg
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_context

from sglang_omni.models.qwen3_omni.components import thinker_kernels
from sglang_omni.models.qwen3_omni.components.thinker_kernels import (
    norm_rope_store,
    supports_norm_rope_store,
)
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.rocm import ROCMOmniPlatform

HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 4
Q_SIZE = NUM_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
MAX_POSITION = 65536
CACHE_ROWS = 4096
MAX_TOKENS = 128
EPS = 1e-6
DTYPE = torch.bfloat16


@pytest.fixture
def published_server_args() -> Iterator[None]:
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        yield


def thinker_attention(device: torch.device) -> SimpleNamespace:
    rotary = get_rope(
        HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=MAX_POSITION,
        base=1000000,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    ).to(device)
    norms = []
    for _ in range(2):
        norm = RMSNorm(HEAD_DIM, eps=EPS).to(device, DTYPE)
        with torch.no_grad():
            norm.weight.normal_(1.0, 0.1)
        norms.append(norm)
    return SimpleNamespace(
        rotary_emb=rotary,
        q_norm=norms[0],
        k_norm=norms[1],
        head_dim=HEAD_DIM,
        num_heads=NUM_HEADS,
        total_num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
    )


def decode_inputs(
    tokens: int, equal_rows: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = torch.randn(tokens, Q_SIZE + 2 * KV_SIZE, device=device)
    qkv *= torch.rand(tokens, 1, device=device) * 8
    position_rows = torch.zeros(3, MAX_TOKENS, dtype=torch.int64, device=device)
    first_row = torch.randint(0, MAX_POSITION, (tokens,), device=device)
    if equal_rows:
        position_rows[:, :tokens] = first_row
    else:
        position_rows[0, :tokens] = first_row
        position_rows[1:, :tokens] = torch.randint(
            0, MAX_POSITION, (2, tokens), device=device
        )
    cache_loc = torch.randperm(CACHE_ROWS - 1, device=device)[:tokens] + 1
    cache_loc[tokens // 2] = 0
    return qkv.to(DTYPE), position_rows[:, :tokens], cache_loc


def sglang_decode(
    attention: SimpleNamespace,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
) -> torch.Tensor:
    q, k, v = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
    q, k = apply_qk_norm(q, k, attention.q_norm, attention.k_norm, HEAD_DIM)
    q, k = attention.rotary_emb.forward_cuda(positions, q, k)
    store_cache(k, v, k_cache, v_cache, cache_loc, size_limit=CACHE_ROWS)
    return q


def fused_decode(
    attention: SimpleNamespace,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
) -> torch.Tensor:
    kv_store = FusedSetKVBufferArg(
        value=qkv[:, Q_SIZE + KV_SIZE :],
        k_buffer=k_cache,
        v_buffer=v_cache,
        cache_loc=cache_loc,
    )
    q, _, _ = norm_rope_store(qkv, positions, attention, kv_store)
    return q


@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
@pytest.mark.parametrize("equal_rows", [True, False])
@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 64])
def test_one_launch_equals_the_norm_rope_and_store_kernels_bit_for_bit(
    tokens: int, equal_rows: bool
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(tokens)
    attention = thinker_attention(device)
    qkv, positions, cache_loc = decode_inputs(tokens, equal_rows, device)
    caches = torch.randn(2, CACHE_ROWS, KV_SIZE, device=device).to(DTYPE)
    reference_qkv, fused_qkv = qkv.clone(), qkv.clone()
    reference_k, reference_v = caches.clone().unbind(0)
    fused_k, fused_v = caches.clone().unbind(0)

    reference_q = sglang_decode(
        attention, reference_qkv, positions, reference_k, reference_v, cache_loc
    )
    fused_q = fused_decode(attention, fused_qkv, positions, fused_k, fused_v, cache_loc)

    assert fused_q.is_contiguous()
    assert torch.equal(fused_q, reference_q)
    assert torch.equal(fused_qkv[:, Q_SIZE:], reference_qkv[:, Q_SIZE:])
    assert torch.equal(fused_k, reference_k)
    assert torch.equal(fused_v, reference_v)
    assert torch.equal(fused_k[0], caches[0, 0])
    assert torch.equal(fused_v[0], caches[1, 0])


@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_one_launch_replays_from_a_cuda_graph_with_new_inputs() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    tokens = 8
    attention = thinker_attention(device)
    qkv, positions, cache_loc = decode_inputs(tokens, True, device)
    k_cache = torch.zeros(CACHE_ROWS, KV_SIZE, device=device, dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)
    attention.rotary_emb._match_cos_sin_cache_dtype(qkv)  # noqa: leading-underscore
    fused_decode(attention, qkv.clone(), positions, k_cache, v_cache, cache_loc)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_q = fused_decode(attention, qkv, positions, k_cache, v_cache, cache_loc)

    new_qkv, new_positions, new_cache_loc = decode_inputs(tokens, True, device)
    qkv.copy_(new_qkv)
    positions.copy_(new_positions)
    cache_loc.copy_(new_cache_loc)
    graph.replay()
    reference_k = torch.zeros_like(k_cache)
    reference_v = torch.zeros_like(v_cache)
    reference_q = sglang_decode(
        attention, new_qkv, new_positions, reference_k, reference_v, new_cache_loc
    )

    assert torch.equal(graph_q, reference_q)
    for row in new_cache_loc.tolist():
        if row != 0:
            assert torch.equal(k_cache[row], reference_k[row])
            assert torch.equal(v_cache[row], reference_v[row])
        else:
            pass


@pytest.mark.accelerator
@pytest.mark.usefixtures("published_server_args")
def test_the_kernel_is_refused_off_cuda_and_under_deterministic_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = thinker_attention(torch.device("cuda"))
    monkeypatch.setattr(thinker_kernels, "current_platform", CUDAOmniPlatform())
    assert supports_norm_rope_store(attention)
    with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
        assert not supports_norm_rope_store(attention)
    monkeypatch.setattr(thinker_kernels, "current_platform", ROCMOmniPlatform())
    assert not supports_norm_rope_store(attention)
