# SPDX-License-Identifier: Apache-2.0
"""GQA equivalence tests for the Qwen3-Omni predictor attention paths."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context

import sglang_omni.models.qwen3_omni.components.talker as talker_module
from sglang_omni.models.qwen3_omni.components.talker import (
    Qwen3OmniMoeTalkerCodePredictor,
    Qwen3OmniTalker,
)
from sglang_omni.platforms import current_platform
from sglang_omni.vendor.sglang.layers import RMSNorm
from tests.unit_test.fixtures.qwen_predictor import (
    TupleLinear,
    build_real_step_predictor_graph_talker,
)

# note (EdwardZhang1108): cpu/fp32 covers the math backend; cuda/bf16 locks the
# production-dtype evidence into CI instead of living only in the PR description.
DEVICE_DTYPE_PARAMS = [
    pytest.param("cpu", torch.float32, id="cpu-fp32"),
    pytest.param(
        "cuda",
        torch.bfloat16,
        id="cuda-bf16",
        marks=[
            pytest.mark.accelerator,
            pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA bf16 variant requires a GPU",
            ),
        ],
    ),
]


def build_gqa_talker(device: torch.device, dtype: torch.dtype) -> Qwen3OmniTalker:
    talker = build_real_step_predictor_graph_talker(device, num_heads=4, num_kv_heads=2)
    attn = talker.code_predictor.model.layers[0].self_attn
    # note (EdwardZhang1108): kv heads > 1, else wrong GQA group order passes by broadcast
    assert attn.num_heads != attn.num_kv_heads and attn.num_kv_heads > 1
    if dtype is not torch.float32:
        attn.qkv_proj.to(dtype)
        attn.o_proj.to(dtype)
        talker.predictor_k_cache = talker.predictor_k_cache.to(dtype)
        talker.predictor_v_cache = talker.predictor_v_cache.to(dtype)
    return talker


def project_q_kv(attn: SimpleNamespace, hidden_states: torch.Tensor):
    """Shared projection: hidden states to per-head q/k/v, mirroring the source."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    qkv, _ = attn.qkv_proj(hidden_states.reshape(-1, hidden_size))
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

    def heads(t: torch.Tensor, num: int) -> torch.Tensor:
        return t.reshape(batch_size, seq_len, num, attn.head_dim).transpose(1, 2)

    return (
        heads(q, attn.num_heads),
        heads(k, attn.num_kv_heads),
        heads(v, attn.num_kv_heads),
    )


def materialized_sdpa(
    attn: SimpleNamespace,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    """Reference attention that materializes KV heads before SDPA."""
    num_kv_groups = attn.num_heads // attn.num_kv_heads
    k = k.repeat_interleave(num_kv_groups, dim=1)
    v = v.repeat_interleave(num_kv_groups, dim=1)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal
    )
    batch_size, _, seq_len, _ = q.shape
    attn_output = attn_output.transpose(1, 2).reshape(
        batch_size * seq_len, attn.num_heads * attn.head_dim
    )
    attn_output, _ = attn.o_proj(attn_output)
    return attn_output.reshape(batch_size, seq_len, -1)


def materialized_kv_direct_attention(
    *,
    attn: SimpleNamespace,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    q, k, v = project_q_kv(attn, hidden_states)
    return materialized_sdpa(attn, q, k, v, is_causal=True)


def materialized_kv_cached_attention(
    *,
    talker: Qwen3OmniTalker,
    attn: SimpleNamespace,
    hidden_states: torch.Tensor,
    batch_size: int,
    cache_len: int,
) -> torch.Tensor:
    q, _, _ = project_q_kv(attn, hidden_states)
    cached_k = talker.predictor_k_cache[0, :batch_size, :, : cache_len + 1, :]
    cached_v = talker.predictor_v_cache[0, :batch_size, :, : cache_len + 1, :]
    return materialized_sdpa(attn, q, cached_k, cached_v, is_causal=False)


@pytest.mark.parametrize("device_name,dtype", DEVICE_DTYPE_PARAMS)
def test_qwen_predictor_direct_attention_gqa_matches_materialized_kv(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    dtype: torch.dtype,
):
    """Direct-path SDPA with enable_gqa must equal materialized KV expansion."""
    monkeypatch.setattr(
        talker_module,
        "apply_qk_norm",
        lambda q, k, **_: (q, k),
    )

    device = torch.device(device_name)
    talker = build_gqa_talker(device, dtype)
    attn = talker.code_predictor.model.layers[0].self_attn

    batch_size, seq_len, hidden_size = 2, 3, 8
    torch.manual_seed(7)
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=dtype
    )
    positions = torch.arange(seq_len, device=device).repeat(batch_size)

    with torch.no_grad():
        actual = Qwen3OmniMoeTalkerCodePredictor.direct_self_attention(
            attn=attn,
            hidden_states=hidden_states,
            positions=positions,
        )
        expected = materialized_kv_direct_attention(
            attn=attn,
            hidden_states=hidden_states,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device_name,dtype", DEVICE_DTYPE_PARAMS)
def test_qwen_predictor_cached_attention_gqa_matches_materialized_kv(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    dtype: torch.dtype,
):
    """Cached-decode SDPA with enable_gqa must equal materialized KV expansion."""
    monkeypatch.setattr(
        talker_module,
        "apply_qk_norm",
        lambda q, k, **_: (q, k),
    )

    device = torch.device(device_name)
    talker = build_gqa_talker(device, dtype)
    attn = talker.code_predictor.model.layers[0].self_attn
    batch_size, hidden_size = 2, 8

    torch.manual_seed(11)
    with torch.no_grad():
        for cache_len in range(3):
            hidden_states = torch.randn(
                batch_size, 1, hidden_size, device=device, dtype=dtype
            )
            positions = torch.full((batch_size,), cache_len, device=device)
            actual = talker.predictor_cached_self_attention(
                layer_idx=0,
                attn=attn,
                hidden_states=hidden_states,
                positions=positions,
                batch_size=batch_size,
                cache_len=cache_len,
            )
            expected = materialized_kv_cached_attention(
                talker=talker,
                attn=attn,
                hidden_states=hidden_states,
                batch_size=batch_size,
                cache_len=cache_len,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


LAYOUT_HEAD_DIM = 128
LAYOUT_NUM_HEADS = 16
LAYOUT_NUM_KV_HEADS = 8
LAYOUT_HIDDEN = 1024
LAYOUT_PREDICTOR_LEN = 17
LAYOUT_MAX_BS = 16
LAYOUT_DTYPE = torch.bfloat16


@pytest.fixture
def published_server_args() -> Iterator[None]:
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        yield


def layout_talker(device: torch.device) -> Qwen3OmniTalker:
    talker = object.__new__(Qwen3OmniTalker)
    positions = torch.arange(LAYOUT_PREDICTOR_LEN, device=device, dtype=torch.long)
    talker.predictor_positions = positions
    talker.predictor_position_rows = (
        positions[:, None].expand(LAYOUT_PREDICTOR_LEN, LAYOUT_MAX_BS).contiguous()
    )
    talker.predictor_pair_positions = positions[:2].repeat(LAYOUT_MAX_BS)
    talker.predictor_k_cache = torch.zeros(
        1,
        LAYOUT_MAX_BS,
        LAYOUT_NUM_KV_HEADS,
        LAYOUT_PREDICTOR_LEN,
        LAYOUT_HEAD_DIM,
        device=device,
        dtype=LAYOUT_DTYPE,
    )
    talker.predictor_v_cache = torch.zeros_like(talker.predictor_k_cache)
    talker.predictor_layer_shape = None
    talker.predictor_o_proj_transposed = False
    talker.predictor_o_proj_weights_t = []
    talker.predictor_exact_add_norm = False
    return talker


def layout_attention(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        q_size=LAYOUT_NUM_HEADS * LAYOUT_HEAD_DIM,
        kv_size=LAYOUT_NUM_KV_HEADS * LAYOUT_HEAD_DIM,
        num_heads=LAYOUT_NUM_HEADS,
        num_kv_heads=LAYOUT_NUM_KV_HEADS,
        head_dim=LAYOUT_HEAD_DIM,
        q_norm=RMSNorm(LAYOUT_HEAD_DIM, eps=1e-6).to(device, LAYOUT_DTYPE),
        k_norm=RMSNorm(LAYOUT_HEAD_DIM, eps=1e-6).to(device, LAYOUT_DTYPE),
        alt_stream=None,
        qkv_proj=TupleLinear(
            LAYOUT_HIDDEN,
            (LAYOUT_NUM_HEADS + 2 * LAYOUT_NUM_KV_HEADS) * LAYOUT_HEAD_DIM,
        ).to(device, LAYOUT_DTYPE),
        o_proj=TupleLinear(LAYOUT_NUM_HEADS * LAYOUT_HEAD_DIM, LAYOUT_HIDDEN).to(
            device, LAYOUT_DTYPE
        ),
        rotary_emb=RotaryEmbedding(
            LAYOUT_HEAD_DIM, LAYOUT_HEAD_DIM, 64, 10000, True, LAYOUT_DTYPE
        ).to(device),
    )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the cuBLAS layouts are compared on CUDA"
)
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_o_proj_on_the_transposed_weight_matches_the_linear(
    batch_size: int, published_server_args: None
) -> None:
    """The (K, N) weight copy changes the cuBLAS kernel, not the bits."""
    del published_server_args
    device = torch.device("cuda")
    talker = layout_talker(device)
    attn = layout_attention(device)
    torch.manual_seed(0)
    hidden = torch.randn(
        batch_size, 1, LAYOUT_HIDDEN, device=device, dtype=LAYOUT_DTYPE
    )
    positions = talker.predictor_position_rows[0, :batch_size]

    def run_attention() -> torch.Tensor:
        return talker.predictor_cached_self_attention(
            layer_idx=0,
            attn=attn,
            hidden_states=hidden,
            positions=positions,
            batch_size=batch_size,
            cache_len=0,
        )

    through_linear = run_attention()
    talker.predictor_o_proj_transposed = True
    talker.predictor_o_proj_weights_t = [attn.o_proj.proj.weight.t().contiguous()]
    assert torch.equal(run_attention(), through_linear)
