"""Distance to fp32 of the Qwen3-TTS predictor's plain and fused (P4) layer paths.

#2413's numerics gate for Qwen3-Omni, on the Qwen3-TTS predictor with the checkpoint's own
weights: the five code predictor layers of Qwen/Qwen3-TTS-12Hz-1.7B-Base loaded into the
real shape fixtures of tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py (P4's tree),
rope at the checkpoint's theta. Over the served sequence (the opening pair, then 14 one
token passes) at each batch size, both bf16 paths run on the same inputs and an fp32
reference runs the same bf16 weights in fp32 with fp32 caches. Prints, per batch size, each
path's relative error to the reference per step (max and mean over steps) and for the K
cache, and the fused over plain ratio, which #2413 bounds at 1.25.

usage: PYTHONPATH=<P4 tree> python predictor_fp32_distance.py [--batches 1 5 16 64]
       [--seeds 3]
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.vendor.sglang.models import apply_qk_norm
from tests.unit_test.qwen3_tts import test_predictor_cuda_graph as fixtures

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PREFIX = "talker.code_predictor.model."


def load_layers(device: torch.device):
    with open(hf_hub_download(MODEL, "config.json")) as handle:
        config = json.load(handle)["talker_config"]
    cp = config["code_predictor_config"]
    assert (cp["hidden_size"], cp["intermediate_size"]) == (
        fixtures.FUSED_HIDDEN,
        fixtures.FUSED_INTERMEDIATE,
    )
    assert (cp["num_attention_heads"], cp["num_key_value_heads"], cp["head_dim"]) == (
        fixtures.FUSED_NUM_HEADS,
        fixtures.FUSED_NUM_KV_HEADS,
        fixtures.FUSED_HEAD_DIM,
    )
    path = hf_hub_download(MODEL, "model.safetensors")
    weights = {}
    with safe_open(path, "pt", device=str(device)) as handle:
        for key in handle.keys():
            if key.startswith(PREFIX):
                weights[key[len(PREFIX) :]] = handle.get_tensor(key).to(fixtures.DTYPE)
    layers = []
    with torch.no_grad():
        for index in range(cp["num_hidden_layers"]):
            p = f"layers.{index}."
            layer = fixtures.real_shape_layer(device)
            attention = layer.self_attn
            attention.qkv_proj.weight.copy_(
                torch.cat([weights[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])
            )
            attention.o_proj.weight.copy_(weights[p + "self_attn.o_proj.weight"])
            attention.q_norm.weight.copy_(weights[p + "self_attn.q_norm.weight"])
            attention.k_norm.weight.copy_(weights[p + "self_attn.k_norm.weight"])
            attention.rotary_emb = RotaryEmbedding(
                fixtures.FUSED_HEAD_DIM,
                fixtures.FUSED_HEAD_DIM,
                64,
                int(cp["rope_theta"]),
                True,
                fixtures.DTYPE,
            ).to(device)
            layer.input_layernorm.weight.copy_(weights[p + "input_layernorm.weight"])
            layer.post_attention_layernorm.weight.copy_(
                weights[p + "post_attention_layernorm.weight"]
            )
            layer.mlp.gate_up_proj.weight.copy_(
                torch.cat(
                    (
                        weights[p + "mlp.gate_proj.weight"],
                        weights[p + "mlp.up_proj.weight"],
                    )
                )
            )
            layer.mlp.down_proj.weight.copy_(weights[p + "mlp.down_proj.weight"])
            layers.append(layer)
        final_norm = fixtures.real_shape_norm(fixtures.FUSED_HIDDEN, device)
        final_norm.weight.copy_(weights["norm.weight"])
    return layers, final_norm, float(cp["rms_norm_eps"])


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def rotate(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    cos, sin = cos_sin[..., None, :half], cos_sin[..., None, half:]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), -1)


class Reference:
    """The layers in fp32 from the same bf16 weights, with fp32 caches laid out
    (layer, row, kv head, slot, head dim)."""

    def __init__(self, layers, final_norm, eps: float, batch_size: int, device) -> None:
        self.layers, self.final_norm, self.eps = layers, final_norm, eps
        self.k_cache = torch.zeros(
            len(layers),
            batch_size,
            fixtures.FUSED_NUM_KV_HEADS,
            fixtures.FUSED_PREDICTOR_LEN,
            fixtures.FUSED_HEAD_DIM,
            device=device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

    def forward(self, token_embeds: torch.Tensor, cache_len: int) -> torch.Tensor:
        batch_size, seq_len, _ = token_embeds.shape
        end = cache_len + seq_len
        heads, kv_heads, head_dim = (
            fixtures.FUSED_NUM_HEADS,
            fixtures.FUSED_NUM_KV_HEADS,
            fixtures.FUSED_HEAD_DIM,
        )
        x = token_embeds.float()
        positions = torch.arange(cache_len, end, device=x.device)
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            cos_sin = attention.rotary_emb.cos_sin_cache[positions].float()
            normed = rmsnorm(x, layer.input_layernorm.weight, self.eps)
            qkv = normed @ attention.qkv_proj.weight.float().t()
            q, k, v = qkv.split(
                [attention.q_size, attention.kv_size, attention.kv_size], -1
            )
            q = q.reshape(batch_size, seq_len, heads, head_dim)
            k = k.reshape(batch_size, seq_len, kv_heads, head_dim)
            v = v.reshape(batch_size, seq_len, kv_heads, head_dim)
            q = rotate(rmsnorm(q, attention.q_norm.weight, self.eps), cos_sin)
            k = rotate(rmsnorm(k, attention.k_norm.weight, self.eps), cos_sin)
            self.k_cache[index, :, :, cache_len:end] = k.transpose(1, 2)
            self.v_cache[index, :, :, cache_len:end] = v.transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                self.k_cache[index, :, :, :end],
                self.v_cache[index, :, :, :end],
                is_causal=seq_len > 1,
                enable_gqa=True,
            )
            attended = attended.transpose(1, 2).reshape(batch_size, seq_len, -1)
            x = x + attended @ attention.o_proj.weight.float().t()
            normed = rmsnorm(x, layer.post_attention_layernorm.weight, self.eps)
            gate, up = (normed @ layer.mlp.gate_up_proj.weight.float().t()).chunk(2, -1)
            x = x + (F.silu(gate) * up) @ layer.mlp.down_proj.weight.float().t()
        return rmsnorm(x, self.final_norm.weight, self.eps)


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference).norm() / reference.norm()).item()


def run_sequence(talker, steps) -> list[torch.Tensor]:
    talker.predictor_k_cache.zero_()
    talker.predictor_v_cache.zero_()
    outputs, cache_len = [], 0
    for step in steps:
        outputs.append(
            talker.predictor_forward_tokens(
                token_embeds=step.clone(), batch_size=step.shape[0], cache_len=cache_len
            ).clone()
        )
        cache_len += step.shape[1]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 5, 16, 64])
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda")
    sglang_model_module.apply_qk_norm = apply_qk_norm
    layers, final_norm, eps = load_layers(device)
    talkers = {
        name: fixtures.real_shape_talker(device, layers, final_norm, fused=fused)
        for name, fused in (("plain", False), ("fused", True))
    }
    print(
        f"{len(layers)} layers from {MODEL}, eps {eps}; errors are relative to the fp32 "
        "reference (norm of the difference over the norm of the reference)"
    )
    print(
        f"{'bs':>3} {'seed':>4} {'plain max':>10} {'fused max':>10} {'plain mean':>11} "
        f"{'fused mean':>11} {'worst step ratio':>17} {'K plain':>9} {'K fused':>9}"
    )
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
        torch.no_grad(),
    ):
        for batch_size in args.batches:
            for seed in range(args.seeds):
                generator = torch.Generator(device=device).manual_seed(seed)
                steps = [
                    torch.randn(
                        batch_size,
                        2,
                        fixtures.FUSED_HIDDEN,
                        device=device,
                        generator=generator,
                    )
                ] + [
                    torch.randn(
                        batch_size,
                        1,
                        fixtures.FUSED_HIDDEN,
                        device=device,
                        generator=generator,
                    )
                    for _ in range(fixtures.FUSED_PREDICTOR_LEN - 3)
                ]
                steps = [step.to(fixtures.DTYPE) for step in steps]
                outputs, k_caches = {}, {}
                for name, talker in talkers.items():
                    outputs[name] = run_sequence(talker, steps)
                    k_caches[name] = (
                        talker.predictor_k_cache[:, :batch_size].transpose(2, 3).clone()
                    )
                reference = Reference(layers, final_norm, eps, batch_size, device)
                errors = {"plain": [], "fused": []}
                cache_len = 0
                for index, step in enumerate(steps):
                    expected = reference.forward(step, cache_len)
                    for name in errors:
                        errors[name].append(
                            relative_error(outputs[name][index], expected)
                        )
                    cache_len += step.shape[1]
                ratio = max(f / p for f, p in zip(errors["fused"], errors["plain"]))
                filled = reference.k_cache[..., :cache_len, :]
                k_error = {
                    name: relative_error(k[..., :cache_len, :], filled)
                    for name, k in k_caches.items()
                }
                print(
                    f"{batch_size:>3} {seed:>4} {max(errors['plain']):>10.5f} {max(errors['fused']):>10.5f} "
                    f"{sum(errors['plain']) / len(steps):>11.5f} {sum(errors['fused']) / len(steps):>11.5f} "
                    f"{ratio:>17.3f} {k_error['plain']:>9.5f} {k_error['fused']:>9.5f}"
                )


if __name__ == "__main__":
    main()
