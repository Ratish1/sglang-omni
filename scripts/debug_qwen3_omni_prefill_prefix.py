# SPDX-License-Identifier: Apache-2.0
"""Replay every token of a held Talker prefix through its first two K/V producers."""

import argparse
import json
from itertools import accumulate
from pathlib import Path

import torch
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.environ import envs
from sglang.srt.layers.activation import silu_and_mul
from sglang.srt.layers.moe.topk import fused_topk
from sglang.srt.layers.rotary_embedding import get_rope
from torch.nn.functional import linear

from scripts.debug_qwen3_omni_peer_shape import read_weights


def cache_rows(
    checkpoint: Path,
    layer_index: int,
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, q_norm, k_norm = read_weights(
        checkpoint,
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ),
        layer_index,
    )
    head_size = q_norm.numel()
    query, key, value = linear(x, torch.cat((q, k, v))).split(
        (q.shape[0], k.shape[0], v.shape[0]), dim=-1
    )
    query = rmsnorm(query.contiguous().view(-1, head_size), q_norm, epsilon)
    key = rmsnorm(key.contiguous().view(-1, head_size), k_norm, epsilon)
    query, key = rotary(
        positions, query.reshape(x.shape[0], -1), key.reshape(x.shape[0], -1)
    )
    return tuple(
        t.contiguous().view(x.shape[0], -1, head_size) for t in (query, key, value)
    )


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: tuple[int, ...]
) -> torch.Tensor:
    starts = (0, *accumulate(lengths))
    count = starts[-1]
    device = q.device
    maximum = max(lengths)
    pages = torch.zeros((len(lengths), maximum), dtype=torch.int32, device=device)
    for index, length in enumerate(lengths):
        pages[index, :length] = torch.arange(
            starts[index], starts[index + 1], dtype=torch.int32, device=device
        )
    sizes = torch.tensor(lengths, dtype=torch.int32, device=device)
    offsets = torch.tensor(starts, dtype=torch.int32, device=device)
    out = flash_attn_with_kvcache(
        q=q[:count],
        k_cache=k.unsqueeze(1),
        v_cache=v.unsqueeze(1),
        page_table=pages,
        cache_seqlens=sizes,
        cu_seqlens_q=offsets,
        cu_seqlens_k_new=offsets,
        max_seqlen_q=maximum,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=True,
        num_splits=0,
        ver=3,
    )
    result = q.new_zeros((q.shape[0], q.shape[1] * q.shape[2]))
    result[:count] = out.flatten(1)
    return result


def row_difference(
    actual: torch.Tensor, native: torch.Tensor
) -> dict[str, int | float]:
    delta = actual.float().flatten(1) - native.cuda().float().flatten(1)
    return {
        "changed_rows": int(delta.ne(0).any(dim=1).sum()),
        "maximum_absolute_difference": float(delta.abs().max()),
    }


def routed_experts(
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    tactics: list[int] | None = None,
) -> torch.Tensor:
    return cutlass_fused_moe(
        input=inputs,
        token_selected_experts=expert_ids,
        token_final_scales=weights,
        fc1_expert_weights=fc1,
        fc2_expert_weights=fc2,
        output_dtype=inputs.dtype,
        output=torch.empty_like(inputs),
        quant_scales=None,
        tune_max_num_tokens=8192,
        activation_type=ActivationType.Swiglu,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
        profile_ids=tactics,
    )[0]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--historical-peer", type=Path, required=True)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--gemm1-tactics", type=int, nargs=2)
    parser.add_argument("--gemm2-tactic", type=int)
    args = parser.parse_args()
    state = torch.load(args.input, weights_only=True)
    serial = torch.load(args.serial, weights_only=True)
    peer = torch.load(args.historical_peer, weights_only=True)
    embeddings = state["embeddings"].cuda()
    held_positions = state["rotary_positions"].cuda()
    assert embeddings.shape[0] == 151 and held_positions.shape == (3, 151)
    text = json.loads((args.checkpoint / "config.json").read_bytes())["talker_config"][
        "text_config"
    ]
    heads = text["head_dim"]
    epsilon = text["rms_norm_eps"]
    rope = get_rope(
        heads,
        rotary_dim=heads,
        max_position=text["max_position_embeddings"],
        base=text["rope_theta"],
        rope_scaling=text["rope_scaling"],
        dtype=embeddings.dtype,
    ).cuda()
    (
        attention_norm,
        attn_out,
        ffn_norm,
        shared_gate,
        shared_up,
        shared_down,
        scalar,
        router,
    ) = read_weights(
        args.checkpoint,
        (
            "input_layernorm",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
            "mlp.shared_expert_gate",
            "mlp.gate",
        ),
    )
    (next_norm,) = read_weights(args.checkpoint, ("input_layernorm",), layer_index=1)
    expert_count = text["num_experts"]
    expert_width = text["moe_intermediate_size"]
    width = embeddings.shape[1]
    packed_fc1 = embeddings.new_empty((expert_count, 2 * expert_width, width))
    packed_fc2 = embeddings.new_empty((expert_count, width, expert_width))
    for expert in range(expert_count):
        gate, up, down = read_weights(
            args.checkpoint,
            tuple(
                f"mlp.experts.{expert}.{part}_proj" for part in ("gate", "up", "down")
            ),
        )
        packed_fc1[expert] = torch.cat((up, gate))
        packed_fc2[expert] = down
    AutoTuner.get().load_configs(str(args.autotune_cache))
    layouts = (
        ((151,), 0, 151),
        ((151,), 0, 160),
        ((151, 145), 0, 320),
        ((146, 151, 121, 136, 171, 118, 112, 138, 140, 178), 1, 1536),
        ((151, *([143] * 10), 142), 0, 1792),
    )
    serial_stages = None
    for lengths, target_rank, physical_count in layouts:
        start = sum(lengths[:target_rank])
        target = slice(start, start + embeddings.shape[0])
        # Neighbor rows exercise kernels and routing, but are not the historical
        # input. Only the saved 151-row target is the comparison subject.
        x = embeddings.roll(1, dims=1).repeat((physical_count + 150) // 151, 1)
        x = x[:physical_count].contiguous()
        x[target] = embeddings
        rotary_positions = torch.zeros(
            (3, physical_count), dtype=torch.long, device=x.device
        )
        rotary_positions[:, target] = held_positions
        normalized = rmsnorm(x, attention_norm, epsilon)
        q, k, v = cache_rows(
            args.checkpoint, 0, normalized, rotary_positions, rope, epsilon
        )
        context = attention(q, k, v, lengths)
        ffn_input = linear(context, attn_out)
        fused_add_rmsnorm(ffn_input, x, ffn_norm, epsilon)
        shared = linear(
            silu_and_mul(linear(ffn_input, torch.cat((shared_gate, shared_up)))),
            shared_down,
        )
        shared = shared * linear(ffn_input, scalar).sigmoid()
        router_logits = linear(ffn_input, router)
        weights, ids = fused_topk(
            ffn_input,
            router_logits,
            text["num_experts_per_tok"],
            text["norm_topk_prob"],
        )
        routed = routed_experts(ffn_input, ids, weights, packed_fc1, packed_fc2)
        block = routed + shared
        next_input = block.clone()
        fused_add_rmsnorm(next_input, x, next_norm, epsilon)
        _, next_k, next_v = cache_rows(
            args.checkpoint, 1, next_input, rotary_positions, rope, epsilon
        )
        stages = {
            "layer_zero_keys": k[target],
            "layer_zero_values": v[target],
            "layer_zero_context": context[target],
            "layer_zero_ffn_input": ffn_input[target],
            "shared_output": shared[target],
            "routed_output": routed[target],
            "layer_zero_output": block[target],
            "layer_one_keys": next_k[target],
            "layer_one_values": next_v[target],
        }
        if physical_count == 160:
            serial_stages = {name: value.clone() for name, value in stages.items()}
        else:
            pass
        if args.gemm1_tactics and physical_count in (1536, 1792):
            assert args.gemm2_tactic is not None
            for gemm1 in args.gemm1_tactics:
                tactics = [gemm1, args.gemm2_tactic]
                kernel_args = (ffn_input, ids, weights, packed_fc1, packed_fc2, tactics)
                result = routed_experts(*kernel_args)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as profile:
                    routed_experts(*kernel_args)
                    torch.cuda.synchronize()
                kernels = sorted(
                    {
                        event.name
                        for event in profile.events()
                        if "gemm" in event.name.lower()
                    }
                )
                print(
                    json.dumps(
                        {
                            "tactic_swap_shape": physical_count,
                            "tactics": tactics,
                            "against_auto": row_difference(
                                result[target], routed[target]
                            ),
                            "against_replayed_160": row_difference(
                                result[target], serial_stages["routed_output"]
                            ),
                            "gemm_kernel_names": kernels,
                        }
                    ),
                    flush=True,
                )
        else:
            pass
        comparisons = (
            {
                name: row_difference(value, serial_stages[name])
                for name, value in stages.items()
            }
            if serial_stages is not None
            else {}
        )
        print(
            json.dumps(
                {
                    "lengths": lengths,
                    "physical_count": physical_count,
                    "against_serial_capture": {
                        "layer_zero_k": row_difference(
                            k[target], serial["key_rows"][0]
                        ),
                        "layer_zero_v": row_difference(
                            v[target], serial["value_rows"][0]
                        ),
                        "layer_one_k": row_difference(
                            next_k[target], serial["key_rows"][1]
                        ),
                        "layer_one_v": row_difference(
                            next_v[target], serial["value_rows"][1]
                        ),
                    },
                    "against_historical_peer_capture": {
                        "layer_one_k": row_difference(
                            next_k[target], peer["key_rows"][1]
                        ),
                        "layer_one_v": row_difference(
                            next_v[target], peer["value_rows"][1]
                        ),
                    },
                    "against_replayed_160_rows": comparisons,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
