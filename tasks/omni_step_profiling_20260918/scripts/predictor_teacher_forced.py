"""Where the Qwen3-TTS predictor's fused layers (P4) and its plain layers differ from fp32,
on the served chain's real inputs, teacher forced.

Inputs: a record_predictor_chain file (talker hidden, layer 0 code and the 15 served codes
per row). Weights: Qwen/Qwen3-TTS-12Hz-1.7B-Base's code predictor, loaded into P4's real
shape fixtures (tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py, so run with
PYTHONPATH=<P4 tree>), with the served plain MLP (SGLang's SiluAndMul, one rounding) in place
of the fixture's two-rounding one. Three arms on the same bf16 inputs: plain (the served
plain path: vendor RMSNorm and fused add-RMSNorm, qkv, apply_qk_norm, rope, SDPA, addmm
o_proj with the residual, SiluAndMul MLP), fused (#2413's launches) and an fp32 reference
(the same bf16 weights in fp32, fp32 caches). Every pass is fed the served code, so the
three arms see the same inputs at every pass.

The plain arm stores K and V through the rope kernel's fused KV store into row views of the
caches, as serving does (predictor_rope_stores_kv). Checks first: each arm's traced loop
reproduces Qwen3TTSTalker.predictor_forward_tokens bit for bit, and whether rope then copy
(the unit fixture's path) gives the same bits as the fused store.

Reports:
  A  per layer and boundary, the propagated chain's relative error to fp32 (plain, fused)
  B  local stages, both arms from the same bf16 stage inputs (the plain chain's real
     activations), output error to fp32 from those inputs: the add and norm into q, k, v;
     o_proj with the residual; the norm into the MLP activation; the final norm
  C  per codebook, the sampling distribution (temperature, top k as served) against the
     reference's: total variation, signed entropy shift, top 1 agreement, logit error

usage: PYTHONPATH=<P4 tree> python predictor_teacher_forced.py CHAIN.pt [--rows 2048]
       [--batch 16] [--temperature 0.9] [--top-k 50]
"""

from __future__ import annotations

import argparse
import collections
import json

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.models.qwen3_omni.components.predictor_kernels import (
    attention_inputs,
    down_add,
    mlp_up,
    o_proj_add,
)
from sglang_omni.models.qwen3_tts.sglang_model import (
    Qwen3TTSTalker,
    predictor_gqa_attention,
)
from sglang_omni.vendor.sglang.models import apply_qk_norm
from tests.unit_test.qwen3_tts import test_predictor_cuda_graph as fixtures

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DTYPE = fixtures.DTYPE


class ServedMLP(torch.nn.Module):
    """The served plain MLP: gate_up, SGLang's SiluAndMul, down."""

    def __init__(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        super().__init__()
        self.gate_up_proj = fixtures.TupleLinear(gate_up.shape[1], gate_up.shape[0])
        self.down_proj = fixtures.TupleLinear(down.shape[1], down.shape[0])
        self.gate_up_proj.proj.weight.data = gate_up
        self.down_proj.proj.weight.data = down
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)[0]))[0]


def load(device: torch.device):
    with open(hf_hub_download(MODEL, "config.json")) as handle:
        talker_config = json.load(handle)["talker_config"]
    cp = talker_config["code_predictor_config"]
    weights = {}
    with safe_open(
        hf_hub_download(MODEL, "model.safetensors"), "pt", device=str(device)
    ) as h:
        for key in h.keys():
            if (
                key.startswith("talker.code_predictor.")
                or key == "talker.model.codec_embedding.weight"
            ):
                weights[key] = h.get_tensor(key).to(DTYPE)
    prefix = "talker.code_predictor.model."
    layers = []
    with torch.no_grad():
        for index in range(cp["num_hidden_layers"]):
            p = f"{prefix}layers.{index}."
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
                DTYPE,
            ).to(device)
            layer.input_layernorm.weight.copy_(weights[p + "input_layernorm.weight"])
            layer.post_attention_layernorm.weight.copy_(
                weights[p + "post_attention_layernorm.weight"]
            )
            layer.mlp = ServedMLP(
                torch.cat(
                    (
                        weights[p + "mlp.gate_proj.weight"],
                        weights[p + "mlp.up_proj.weight"],
                    )
                ).contiguous(),
                weights[p + "mlp.down_proj.weight"].contiguous(),
            )
            layers.append(layer)
        final_norm = fixtures.real_shape_norm(fixtures.FUSED_HIDDEN, device)
        final_norm.weight.copy_(weights[prefix + "norm.weight"])
    groups = talker_config["num_code_groups"]
    chain = {
        "projection_weight": weights[
            "talker.code_predictor.small_to_mtp_projection.weight"
        ],
        "projection_bias": weights[
            "talker.code_predictor.small_to_mtp_projection.bias"
        ],
        "layer0_embedding": weights["talker.model.codec_embedding.weight"],
        "codec_embeddings": [
            weights[f"{prefix}codec_embedding.{i}.weight"] for i in range(groups - 1)
        ],
        "lm_heads": [
            weights[f"talker.code_predictor.lm_head.{i}.weight"]
            for i in range(groups - 1)
        ],
    }
    return layers, final_norm, float(cp["rms_norm_eps"]), chain


def rel(actual: torch.Tensor, reference: torch.Tensor) -> float:
    reference = reference.float()
    return ((actual.float() - reference).norm() / reference.norm()).item()


def rmsnorm32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def rotate32(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    cos, sin = cos_sin[..., None, :half], cos_sin[..., None, half:]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), -1)


class Fp32Chain:
    """The predictor layers in fp32 from the bf16 weights, fp32 caches (layer, row, kv
    head, slot, head dim); trace() records the same boundaries as the bf16 arms."""

    def __init__(self, layers, final_norm, eps, batch_size, device) -> None:
        self.layers, self.final_norm, self.eps = layers, final_norm, eps
        shape = (
            len(layers),
            batch_size,
            fixtures.FUSED_NUM_KV_HEADS,
            fixtures.FUSED_PREDICTOR_LEN,
            fixtures.FUSED_HEAD_DIM,
        )
        self.k_cache = torch.zeros(shape, device=device)
        self.v_cache = torch.zeros(shape, device=device)

    def qkv(self, layer, x, positions):
        attention = layer.self_attn
        batch_size, seq_len, _ = x.shape
        cos_sin = attention.rotary_emb.cos_sin_cache[positions].float()
        normed = rmsnorm32(x, layer.input_layernorm.weight, self.eps)
        q, k, v = (normed @ attention.qkv_proj.weight.float().t()).split(
            [attention.q_size, attention.kv_size, attention.kv_size], -1
        )
        q = q.reshape(batch_size, seq_len, -1, fixtures.FUSED_HEAD_DIM)
        k = k.reshape(batch_size, seq_len, -1, fixtures.FUSED_HEAD_DIM)
        v = v.reshape(batch_size, seq_len, -1, fixtures.FUSED_HEAD_DIM)
        q = rotate32(rmsnorm32(q, attention.q_norm.weight, self.eps), cos_sin)
        k = rotate32(rmsnorm32(k, attention.k_norm.weight, self.eps), cos_sin)
        return q, k, v

    def forward(self, token_embeds: torch.Tensor, cache_len: int, trace=None):
        batch_size, seq_len, _ = token_embeds.shape
        end = cache_len + seq_len
        positions = torch.arange(cache_len, end, device=token_embeds.device)
        x = token_embeds.float()
        for index, layer in enumerate(self.layers):
            q, k, v = self.qkv(layer, x, positions)
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
            x = x + attended @ layer.self_attn.o_proj.weight.float().t()
            if trace is not None:
                trace.append(("q", index, q))
                trace.append(("k", index, k))
                trace.append(("v", index, v))
                trace.append(("attention", index, attended))
                trace.append(("residual after o_proj", index, x))
            else:
                pass
            normed = rmsnorm32(x, layer.post_attention_layernorm.weight, self.eps)
            gate, up = (normed @ layer.mlp.gate_up_proj.weight.float().t()).chunk(2, -1)
            activated = F.silu(gate) * up
            x = x + activated @ layer.mlp.down_proj.weight.float().t()
            if trace is not None:
                trace.append(("activated", index, activated))
                trace.append(("residual after down", index, x))
            else:
                pass
        return rmsnorm32(x, self.final_norm.weight, self.eps)


def plain_traced(talker, token_embeds, batch_size, cache_len, trace):
    """Qwen3TTSTalker.predictor_forward_tokens' plain branch, op for op, recording the
    boundaries."""
    num_tokens, hidden_size = token_embeds.shape[1:]
    if num_tokens == 1:
        positions = talker.predictor_position_rows[cache_len, :batch_size]
        cache_slots = talker.predictor_cache_slots[cache_len, :batch_size]
    else:
        positions = talker.predictor_pair_positions[: 2 * batch_size]
        cache_slots = talker.predictor_pair_cache_slots[: 2 * batch_size]
    num_rows = batch_size * num_tokens
    residual = token_embeds.reshape(num_rows, 1, hidden_size)
    mlp_out = None
    shape = (batch_size, num_tokens, -1, fixtures.FUSED_HEAD_DIM)
    for index, layer in enumerate(talker.code_predictor.model.layers):
        if mlp_out is None:
            normed = layer.input_layernorm(residual.reshape(-1, hidden_size))
        else:
            normed, residual = layer.input_layernorm(
                mlp_out, residual.reshape(-1, hidden_size)
            )
            residual = residual.reshape(num_rows, 1, hidden_size)
            trace.append(
                (
                    "residual after down",
                    index - 1,
                    residual.reshape(batch_size, num_tokens, -1).clone(),
                )
            )
        attn_input = talker.predictor_cached_self_attention(
            layer_idx=index,
            attn=layer.self_attn,
            hidden_states=normed.reshape(batch_size, num_tokens, hidden_size),
            positions=positions,
            cache_slots=cache_slots,
            cache_len=cache_len,
        )
        end = cache_len + num_tokens
        k = talker.predictor_k_cache[index, :batch_size, cache_len:end]
        v = talker.predictor_v_cache[index, :batch_size, cache_len:end]
        trace.append(("k", index, k.clone()))
        trace.append(("v", index, v.clone()))
        trace.append(
            ("attention", index, attn_input.reshape(batch_size, num_tokens, -1).clone())
        )
        residual = Qwen3TTSTalker.predictor_o_proj_add_residual(
            layer.self_attn.o_proj, attn_input, residual
        )
        trace.append(
            (
                "residual after o_proj",
                index,
                residual.reshape(batch_size, num_tokens, -1).clone(),
            )
        )
        normed = layer.post_attention_layernorm(residual.reshape(-1, hidden_size))
        gate_up = layer.mlp.gate_up_proj(normed)[0]
        activated = layer.mlp.act_fn(gate_up)
        trace.append(
            ("activated", index, activated.reshape(batch_size, num_tokens, -1).clone())
        )
        mlp_out = layer.mlp.down_proj(activated)[0]
    normed, residual = talker.code_predictor.model.norm(
        mlp_out, residual.reshape(-1, hidden_size)
    )
    trace.append(
        (
            "residual after down",
            index,
            residual.reshape(batch_size, num_tokens, -1).clone(),
        )
    )
    return normed.reshape(batch_size, num_tokens, hidden_size)


def fused_traced(talker, token_embeds, batch_size, cache_len, trace):
    """Qwen3TTSTalker.predictor_forward_tokens_fused, launch for launch, recording the
    boundaries."""
    num_tokens, hidden_size = token_embeds.shape[1:]
    if num_tokens == 1:
        positions = talker.predictor_position_rows[cache_len, :batch_size]
    else:
        positions = talker.predictor_pair_positions[: 2 * batch_size]
    shape = talker.predictor_layer_shape
    rows = batch_size * num_tokens
    end = cache_len + num_tokens
    hidden = token_embeds.reshape(rows, hidden_size)
    residual = talker.predictor_residual[:rows]
    q_out = talker.predictor_q_buffer[:rows]
    activated = talker.predictor_activated[:rows]
    for index, layer in enumerate(talker.code_predictor.model.layers):
        attention = layer.self_attn
        k_cache = talker.predictor_k_cache[index, :batch_size]
        v_cache = talker.predictor_v_cache[index, :batch_size]
        attention_inputs(
            x=hidden,
            tokens_per_row=num_tokens,
            norm=layer.input_layernorm,
            attention=attention,
            q_out=q_out,
            positions=positions,
            k_cache=k_cache,
            v_cache=v_cache,
            partials=talker.predictor_partials,
            sum_sq_partials=talker.predictor_sum_sq_partials,
            counters=talker.predictor_tile_counters,
            shape=shape,
        )
        trace.append(
            ("q", index, q_out.view(batch_size, num_tokens, -1, shape.head_dim).clone())
        )
        trace.append(("k", index, k_cache[:, cache_len:end].clone()))
        trace.append(("v", index, v_cache[:, cache_len:end].clone()))
        attention_output = predictor_gqa_attention(
            q_out.view(
                batch_size, num_tokens, shape.num_q_heads, shape.head_dim
            ).transpose(1, 2),
            k_cache[:, :end].transpose(1, 2),
            v_cache[:, :end].transpose(1, 2),
            num_heads=shape.num_q_heads,
            num_key_value_heads=shape.num_kv_heads,
            is_causal=num_tokens > 1,
        )
        trace.append(
            (
                "attention",
                index,
                attention_output.transpose(1, 2)
                .reshape(batch_size, num_tokens, -1)
                .clone(),
            )
        )
        o_proj_add(
            attention_output=attention_output,
            tokens_per_row=num_tokens,
            weight=attention.o_proj.weight,
            residual_in=hidden,
            residual_out=residual,
            partials=talker.predictor_partials,
            counters=talker.predictor_tile_counters,
            shape=shape,
        )
        trace.append(
            (
                "residual after o_proj",
                index,
                residual.view(batch_size, num_tokens, -1).clone(),
            )
        )
        hidden = residual
        mlp_up(
            residual=residual,
            norm=layer.post_attention_layernorm,
            weight=layer.mlp.gate_up_proj.weight,
            activated=activated,
            shape=shape,
        )
        trace.append(
            ("activated", index, activated.view(batch_size, num_tokens, -1).clone())
        )
        down_add(
            activated=activated,
            weight=layer.mlp.down_proj.weight,
            residual=residual,
            partials=talker.predictor_partials,
            counters=talker.predictor_tile_counters,
            shape=shape,
        )
        trace.append(
            (
                "residual after down",
                index,
                residual.view(batch_size, num_tokens, -1).clone(),
            )
        )
    normed = talker.code_predictor.model.norm(residual)
    return normed.reshape(batch_size, num_tokens, hidden_size)


def sampling_distribution(logits: torch.Tensor, temperature: float, top_k: int):
    scores = logits.float() / temperature
    values, indices = torch.topk(scores, top_k, dim=-1)
    probs = torch.zeros_like(scores)
    probs.scatter_(1, indices, torch.softmax(values, dim=-1))
    return probs


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chain")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda")
    sglang_model_module.apply_qk_norm = apply_qk_norm
    layers, final_norm, eps, chain = load(device)
    talkers = {
        name: fixtures.real_shape_talker(device, layers, final_norm, fused=fused)
        for name, fused in (("plain", False), ("fused", True))
    }
    plain = talkers["plain"]
    plain.predictor_k_rows = [
        c.view(-1, c.shape[-2] * c.shape[-1]) for c in plain.predictor_k_cache
    ]
    plain.predictor_v_rows = [
        c.view(-1, c.shape[-2] * c.shape[-1]) for c in plain.predictor_v_cache
    ]
    plain.predictor_rope_stores_kv = True
    data = torch.load(args.chain)
    total = min(args.rows, data["layer0_codes"].shape[0]) // args.batch * args.batch
    print(f"{total} recorded rows, batch {args.batch}, {len(layers)} layers, eps {eps}")
    print(
        "served batch sizes in the recording:",
        collections.Counter(data["batch_size"].tolist()).most_common(8),
    )

    boundary = collections.defaultdict(lambda: collections.defaultdict(list))
    local = collections.defaultdict(lambda: collections.defaultdict(list))
    per_code = collections.defaultdict(lambda: collections.defaultdict(list))
    checked = False
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
        torch.no_grad(),
    ):
        for start in range(0, total, args.batch):
            rows = slice(start, start + args.batch)
            b = args.batch
            hidden = data["talker_hidden"][rows].to(device, DTYPE).view(b, 1, -1)
            code0 = data["layer0_codes"][rows].to(device)
            codes = data["codes"][rows].to(device)
            embed0 = F.embedding(code0, chain["layer0_embedding"]).view(b, 1, -1)
            steps = [
                F.linear(
                    torch.cat((hidden, embed0), 1),
                    chain["projection_weight"],
                    chain["projection_bias"],
                )
            ]
            for j in range(len(chain["lm_heads"]) - 1):
                embed = F.embedding(codes[:, j], chain["codec_embeddings"][j]).view(
                    b, 1, -1
                )
                steps.append(
                    F.linear(
                        embed, chain["projection_weight"], chain["projection_bias"]
                    )
                )
            for talker in talkers.values():
                talker.predictor_k_cache.zero_()
                talker.predictor_v_cache.zero_()
            reference = Fp32Chain(layers, final_norm, eps, b, device)
            cache_len = 0
            for j, step in enumerate(steps):
                traces = {"plain": [], "fused": [], "fp32": []}
                outs = {
                    "plain": plain_traced(
                        talkers["plain"], step.clone(), b, cache_len, traces["plain"]
                    ),
                    "fused": fused_traced(
                        talkers["fused"], step.clone(), b, cache_len, traces["fused"]
                    ),
                }
                ref_out = reference.forward(step, cache_len, traces["fp32"])
                if not checked:
                    plain = talkers["plain"]
                    kc, vc = (
                        plain.predictor_k_cache.clone(),
                        plain.predictor_v_cache.clone(),
                    )
                    plain.predictor_rope_stores_kv = False
                    copied = plain.predictor_forward_tokens(
                        token_embeds=step.clone(), batch_size=b, cache_len=cache_len
                    )
                    plain.predictor_rope_stores_kv = True
                    print(
                        "check: rope then copy against the fused store, output equal "
                        f"{torch.equal(copied, outs['plain'])}, K equal "
                        f"{torch.equal(plain.predictor_k_cache, kc)}"
                    )
                    plain.predictor_k_cache.copy_(kc)
                    plain.predictor_v_cache.copy_(vc)
                    for name, talker in talkers.items():
                        kc = talker.predictor_k_cache.clone()
                        vc = talker.predictor_v_cache.clone()
                        again = talker.predictor_forward_tokens(
                            token_embeds=step.clone(), batch_size=b, cache_len=cache_len
                        )
                        assert torch.equal(
                            again, outs[name]
                        ), f"{name} trace differs from predictor_forward_tokens"
                        talker.predictor_k_cache.copy_(kc)
                        talker.predictor_v_cache.copy_(vc)
                    checked = True
                    print(
                        "check: both traced loops equal predictor_forward_tokens bit for bit"
                    )
                else:
                    pass
                ref_by = {(kind, i): t for kind, i, t in traces["fp32"]}
                for name in ("plain", "fused"):
                    for kind, i, t in traces[name]:
                        if (kind, i) not in ref_by:
                            continue
                        r = ref_by[(kind, i)]
                        boundary[(kind, i)][name].append(rel(t.reshape(r.shape), r))
                last = slice(1, 2) if j == 0 else slice(0, 1)
                head = chain["lm_heads"][j]
                ref_logits = ref_out[:, last].reshape(b, -1) @ head.float().t()
                ref_probs = sampling_distribution(
                    ref_logits, args.temperature, args.top_k
                )
                for name in ("plain", "fused"):
                    logits = F.linear(outs[name][:, last].reshape(b, -1), head)
                    probs = sampling_distribution(logits, args.temperature, args.top_k)
                    per_code[j][f"{name} logit err"].append(rel(logits, ref_logits))
                    per_code[j][f"{name} tv"].append(
                        0.5 * (probs - ref_probs).abs().sum(-1).mean().item()
                    )
                    per_code[j][f"{name} dH"].append(
                        (entropy(probs) - entropy(ref_probs)).mean().item()
                    )
                    per_code[j][f"{name} top1"].append(
                        (logits.argmax(-1) == ref_logits.argmax(-1))
                        .float()
                        .mean()
                        .item()
                    )
                cache_len += step.shape[1]

    def mean(values):
        return sum(values) / len(values)

    print("\nA. propagated chain: mean relative error to fp32 over passes, per layer")
    print(
        f"{'boundary':24s} {'layer':>5} {'plain':>9} {'fused':>9} {'fused/plain':>11}"
    )
    for (kind, i), arms in sorted(
        boundary.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if "plain" in arms and "fused" in arms:
            p, f = mean(arms["plain"]), mean(arms["fused"])
            print(f"{kind:24s} {i:>5} {p:>9.5f} {f:>9.5f} {f / p:>11.3f}")
        else:
            only = next(iter(arms))
            print(f"{kind:24s} {i:>5} {only}: {mean(arms[only]):.5f}")
    print("\nC. per codebook (served temperature and top k): plain | fused")
    print(
        f"{'code':>4} {'logit err':>21} {'total variation':>21} {'entropy shift':>23} {'top1 = fp32':>19}"
    )
    for j in sorted(per_code):
        m = {key: mean(v) for key, v in per_code[j].items()}
        print(
            f"{j + 1:>4} {m['plain logit err']:>10.5f} {m['fused logit err']:>10.5f} "
            f"{m['plain tv']:>10.5f} {m['fused tv']:>10.5f} "
            f"{m['plain dH']:>+11.5f} {m['fused dH']:>+11.5f} "
            f"{m['plain top1']:>9.4f} {m['fused top1']:>9.4f}"
        )
    totals = {
        key: mean([mean(per_code[j][key]) for j in per_code]) for key in per_code[0]
    }
    print(
        f"mean {totals['plain logit err']:>10.5f} {totals['fused logit err']:>10.5f} "
        f"{totals['plain tv']:>10.5f} {totals['fused tv']:>10.5f} "
        f"{totals['plain dH']:>+11.5f} {totals['fused dH']:>+11.5f} "
        f"{totals['plain top1']:>9.4f} {totals['fused top1']:>9.4f}"
    )


if __name__ == "__main__":
    main()
