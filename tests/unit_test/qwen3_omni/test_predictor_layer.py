# SPDX-License-Identifier: Apache-2.0
"""The fused predictor layer against the plain path and an fp32 reference."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from torch import nn

from sglang_omni.models.qwen3_omni.components.predictor_layer import (
    BLOCK_K,
    BLOCK_N,
    PredictorLayerShape,
    counters_numel,
    partials_numel,
    resolve_predictor_layer_shape,
    split_count,
    sum_sq_partials_numel,
)
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.platforms import current_platform
from tests.unit_test.fixtures.qwen_predictor import TupleLinear

HIDDEN = 1024
HEAD_DIM = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
INTERMEDIATE = 3072
NUM_LAYERS = 5
NUM_CODE_GROUPS = 16
PREDICTOR_LEN = NUM_CODE_GROUPS + 1
MAX_BS = 32
EPS = 1e-6
DTYPE = torch.bfloat16
accelerator = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the fused layer runs on CUDA only"
)


class PlainLinear(TupleLinear):
    """A tuple linear that looks unquantized to the fused layer's resolver."""

    quant_method = UnquantizedLinearMethod()
    bias = None


class SwiGLU(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.gate_up_proj = PlainLinear(HIDDEN, 2 * INTERMEDIATE).to(device, DTYPE)
        self.down_proj = PlainLinear(INTERMEDIATE, HIDDEN).to(device, DTYPE)
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states, _ = self.down_proj(self.act_fn(gate_up))
        return hidden_states


def norm_with_random_scale(device: torch.device, size: int) -> RMSNorm:
    norm = RMSNorm(size, eps=EPS).to(device, DTYPE)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
    return norm


def build_layer(device: torch.device) -> SimpleNamespace:
    attention = SimpleNamespace(
        hidden_size=HIDDEN,
        q_size=NUM_HEADS * HEAD_DIM,
        kv_size=NUM_KV_HEADS * HEAD_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        q_norm=norm_with_random_scale(device, HEAD_DIM),
        k_norm=norm_with_random_scale(device, HEAD_DIM),
        alt_stream=None,
        qkv_proj=PlainLinear(HIDDEN, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM).to(
            device, DTYPE
        ),
        o_proj=PlainLinear(NUM_HEADS * HEAD_DIM, HIDDEN).to(device, DTYPE),
        rotary_emb=RotaryEmbedding(HEAD_DIM, HEAD_DIM, 64, 10000, True, DTYPE).to(
            device
        ),
    )
    return SimpleNamespace(
        self_attn=attention,
        mlp=SwiGLU(device),
        input_layernorm=norm_with_random_scale(device, HIDDEN),
        post_attention_layernorm=norm_with_random_scale(device, HIDDEN),
    )


def build_talker(device: torch.device, seed: int) -> Qwen3OmniTalker:
    """A talker whose predictor layers are real-shape modules with seeded weights."""
    torch.manual_seed(seed)
    talker = object.__new__(Qwen3OmniTalker)
    talker.code_predictor = SimpleNamespace(
        model=SimpleNamespace(
            layers=[build_layer(device) for _ in range(NUM_LAYERS)],
            norm=norm_with_random_scale(device, HIDDEN),
        )
    )
    positions = torch.arange(PREDICTOR_LEN, device=device, dtype=torch.long)
    talker.predictor_positions = positions
    talker.predictor_position_rows = (
        positions[:, None].expand(PREDICTOR_LEN, MAX_BS).contiguous()
    )
    talker.predictor_pair_positions = positions[:2].repeat(MAX_BS)
    talker.predictor_k_cache = torch.zeros(
        NUM_LAYERS,
        MAX_BS,
        NUM_KV_HEADS,
        PREDICTOR_LEN,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
    )
    talker.predictor_v_cache = torch.zeros_like(talker.predictor_k_cache)
    talker.predictor_o_proj_transposed = False
    talker.predictor_o_proj_weights_t = []
    talker.predictor_exact_add_norm = True
    talker.predictor_layer_shape = None
    talker.predictor_q_buffer = torch.zeros(
        2 * MAX_BS, NUM_HEADS * HEAD_DIM, device=device, dtype=DTYPE
    )
    talker.predictor_residual = torch.zeros(
        2 * MAX_BS, HIDDEN, device=device, dtype=DTYPE
    )
    talker.predictor_activated = torch.zeros(
        2 * MAX_BS, INTERMEDIATE, device=device, dtype=DTYPE
    )
    talker.predictor_partials = torch.zeros(0, device=device)
    talker.predictor_sum_sq_partials = torch.zeros(0, device=device)
    talker.predictor_tile_counters = torch.zeros(0, device=device, dtype=torch.int32)
    return talker


def fuse(talker: Qwen3OmniTalker) -> Qwen3OmniTalker:
    device = talker.predictor_k_cache.device
    talker.predictor_layer_shape = resolve_predictor_layer_shape(
        talker.code_predictor, PREDICTOR_LEN, device
    )
    shape = talker.predictor_layer_shape
    assert shape is not None
    talker.predictor_partials = torch.zeros(
        partials_numel(shape, 2 * MAX_BS), device=device
    )
    talker.predictor_sum_sq_partials = torch.zeros(
        sum_sq_partials_numel(shape, 2 * MAX_BS), device=device
    )
    talker.predictor_tile_counters = torch.zeros(
        counters_numel(shape), device=device, dtype=torch.int32
    )
    return talker


def reset_caches(talker: Qwen3OmniTalker) -> None:
    talker.predictor_k_cache.zero_()
    talker.predictor_v_cache.zero_()


@pytest.fixture
def published_server_args() -> Iterator[None]:
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        yield


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()


def rotate(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    half = HEAD_DIM // 2
    cos, sin = cos_sin[..., None, :half], cos_sin[..., None, half:]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), -1)


class Reference:
    """The predictor layers in fp32 from the same bf16 weights, with fp32 caches."""

    def __init__(self, talker: Qwen3OmniTalker, batch_size: int) -> None:
        self.layers = talker.code_predictor.model.layers
        self.final_norm = talker.code_predictor.model.norm
        self.k_cache = torch.zeros(
            NUM_LAYERS,
            batch_size,
            NUM_KV_HEADS,
            PREDICTOR_LEN,
            HEAD_DIM,
            device=talker.predictor_k_cache.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

    def forward(self, token_embeds: torch.Tensor, cache_len: int) -> torch.Tensor:
        batch_size, seq_len, _ = token_embeds.shape
        end = cache_len + seq_len
        x = token_embeds.float()
        positions = torch.arange(cache_len, end, device=x.device)
        for layer_idx, layer in enumerate(self.layers):
            attention = layer.self_attn
            cos_sin = attention.rotary_emb.cos_sin_cache[positions].float()
            normed = rmsnorm(x, layer.input_layernorm.weight)
            qkv = normed @ attention.qkv_proj.weight.float().t()
            q, k, v = qkv.split(
                [attention.q_size, attention.kv_size, attention.kv_size], -1
            )
            q = q.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
            k = k.reshape(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
            v = v.reshape(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
            q = rotate(rmsnorm(q, attention.q_norm.weight), cos_sin)
            k = rotate(rmsnorm(k, attention.k_norm.weight), cos_sin)
            self.k_cache[layer_idx, :, :, cache_len:end] = k.transpose(1, 2)
            self.v_cache[layer_idx, :, :, cache_len:end] = v.transpose(1, 2)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                self.k_cache[layer_idx, :, :, :end],
                self.v_cache[layer_idx, :, :, :end],
                is_causal=seq_len > 1,
                enable_gqa=True,
            )
            attended = attended.transpose(1, 2).reshape(batch_size, seq_len, -1)
            x = x + attended @ attention.o_proj.weight.float().t()
            normed = rmsnorm(x, layer.post_attention_layernorm.weight)
            gate, up = (normed @ layer.mlp.gate_up_proj.weight.float().t()).chunk(2, -1)
            x = (
                x
                + (torch.nn.functional.silu(gate) * up)
                @ layer.mlp.down_proj.weight.float().t()
            )
        return rmsnorm(x, self.final_norm.weight)


def predictor_inputs(
    device: torch.device, batch_size: int, seed: int
) -> list[torch.Tensor]:
    """The opening pair then one token per codebook, as the predictor feeds them."""
    generator = torch.Generator(device=device).manual_seed(seed)
    steps = [torch.randn(batch_size, 2, HIDDEN, device=device, generator=generator)]
    steps += [
        torch.randn(batch_size, 1, HIDDEN, device=device, generator=generator)
        for _ in range(NUM_CODE_GROUPS - 2)
    ]
    return [step.to(DTYPE) for step in steps]


def run_sequence(
    talker: Qwen3OmniTalker, steps: list[torch.Tensor]
) -> list[torch.Tensor]:
    reset_caches(talker)
    outputs = []
    cache_len = 0
    with torch.no_grad():
        for step in steps:
            # note (ratish): the plain path's residual chain writes the first sum
            # into the tokens it was given, as the predictor's scratch allows.
            outputs.append(
                talker.predictor_forward_tokens(
                    token_embeds=step.clone(),
                    batch_size=step.shape[0],
                    cache_len=cache_len,
                ).clone()
            )
            cache_len += step.shape[1]
    return outputs


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference).norm() / reference.norm()).item()


@accelerator
@pytest.mark.accelerator
@pytest.mark.parametrize("batch_size", [1, 3, 12, 32])
def test_fused_layer_tracks_the_reference_like_the_plain_path(
    batch_size: int, published_server_args: None
) -> None:
    """Over a whole predictor sequence the fused path's distance to fp32 stays within
    a quarter of the plain path's at every step, in the output and in the caches."""
    del published_server_args
    device = torch.device("cuda")
    talker = build_talker(device, seed=3)
    steps = predictor_inputs(device, batch_size, seed=4)
    plain = run_sequence(talker, steps)
    plain_k = talker.predictor_k_cache[:, :batch_size].clone()
    fused = run_sequence(fuse(talker), steps)
    fused_k = talker.predictor_k_cache[:, :batch_size].clone()
    reference = Reference(talker, batch_size)
    cache_len = 0
    with torch.no_grad():
        for step, plain_out, fused_out in zip(steps, plain, fused):
            expected = reference.forward(step, cache_len)
            plain_error = relative_error(plain_out, expected)
            fused_error = relative_error(fused_out, expected)
            assert fused_error <= 1.25 * plain_error + 1e-4, (
                cache_len,
                plain_error,
                fused_error,
            )
            cache_len += step.shape[1]
    assert relative_error(fused_k, reference.k_cache) <= 1.25 * relative_error(
        plain_k, reference.k_cache
    )


@accelerator
@pytest.mark.accelerator
def test_fused_layer_is_deterministic_and_batch_invariant(
    published_server_args: None,
) -> None:
    del published_server_args
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=5))
    steps = predictor_inputs(device, 12, seed=6)
    first = run_sequence(talker, steps)
    k_first = talker.predictor_k_cache.clone()
    second = run_sequence(talker, steps)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(k_first, talker.predictor_k_cache)
    alone = run_sequence(talker, [step[:1] for step in steps])
    assert all(torch.equal(a[:1], b) for a, b in zip(first, alone))
    assert torch.equal(k_first[:, :1], talker.predictor_k_cache[:, :1])


@accelerator
@pytest.mark.accelerator
def test_fused_opening_pair_matches_two_single_token_passes(
    published_server_args: None,
) -> None:
    del published_server_args
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=7))
    tokens = predictor_inputs(device, 3, seed=8)[0]
    paired = run_sequence(talker, [tokens])[0]
    paired_k = talker.predictor_k_cache[:, :3, :, :2].clone()
    singles = run_sequence(talker, [tokens[:, 0:1], tokens[:, 1:2]])
    torch.testing.assert_close(paired[:, 1:2], singles[1])
    assert torch.equal(paired_k, talker.predictor_k_cache[:, :3, :, :2])


@accelerator
@pytest.mark.accelerator
def test_fused_layer_replays_from_a_cuda_graph(published_server_args: None) -> None:
    del published_server_args
    device = torch.device("cuda")
    talker = fuse(build_talker(device, seed=9))
    step = predictor_inputs(device, 4, seed=10)[1]
    eager = run_sequence(talker, [step])[0]
    reset_caches(talker)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        talker.predictor_forward_tokens(token_embeds=step, batch_size=4, cache_len=0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    reset_caches(talker)
    with torch.cuda.graph(graph), torch.no_grad():
        replayed = talker.predictor_forward_tokens(
            token_embeds=step, batch_size=4, cache_len=0
        )
    reset_caches(talker)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, eager)


@accelerator
@pytest.mark.accelerator
def test_resolver_reads_the_layer_shape_and_the_split_from_the_device() -> None:
    device = torch.device("cuda")
    talker = build_talker(device, seed=11)
    shape = resolve_predictor_layer_shape(talker.code_predictor, PREDICTOR_LEN, device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert shape == PredictorLayerShape(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_q_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        split_qkv=split_count(
            NUM_HEADS + 2 * NUM_KV_HEADS, HIDDEN // BLOCK_K, sm_count
        ),
        split_hidden=min(
            split_count(HIDDEN // BLOCK_N, NUM_HEADS, sm_count),
            split_count(HIDDEN // BLOCK_N, INTERMEDIATE // BLOCK_K, sm_count),
        ),
    )


@accelerator
@pytest.mark.accelerator
def test_resolver_keeps_the_plain_path_for_a_quantized_projection() -> None:
    device = torch.device("cuda")
    talker = build_talker(device, seed=12)
    talker.code_predictor.model.layers[2].mlp.down_proj.quant_method = SimpleNamespace()
    assert (
        resolve_predictor_layer_shape(talker.code_predictor, PREDICTOR_LEN, device)
        is None
    )


def test_resolver_keeps_the_plain_path_off_cuda() -> None:
    predictor = SimpleNamespace(model=SimpleNamespace(layers=[]))
    assert (
        resolve_predictor_layer_shape(predictor, PREDICTOR_LEN, torch.device("cpu"))
        is None
    )


def test_split_count_lands_the_program_count_nearest_the_sms() -> None:
    assert split_count(32, 16, 132) == 4
    assert split_count(32, 24, 132) == 4
    assert split_count(32, 24, 264) == 8
    assert split_count(32, 12, 132) == 4
    assert split_count(32, 6, 132) == 2
    assert split_count(128, 8, 132) == 1
