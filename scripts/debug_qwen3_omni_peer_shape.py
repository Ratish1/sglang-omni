# SPDX-License-Identifier: Apache-2.0
"""Bisect layer-zero attention and expert projections with held native operands."""

import argparse
import json
from itertools import accumulate
from pathlib import Path

import torch
from safetensors import safe_open
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.layers.activation import silu_and_mul
from torch.nn import functional

LAYER_PREFIX = "talker.model.layers.0"
TOKEN_COUNTS = (151, 160, 296, 320, 1536, 1723, 1792)


def read_weights(checkpoint: Path, names: tuple[str, ...]) -> list[torch.Tensor]:
    weight_index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    tensors = []
    for name in names:
        key = f"{LAYER_PREFIX}.{name}.weight"
        with safe_open(checkpoint / weight_index[key], framework="pt") as shard:
            tensors.append(shard.get_tensor(key).cuda())
    return tensors


def report_difference(actual: torch.Tensor, native: torch.Tensor) -> dict:
    delta = actual.flatten().float() - native.flatten().cuda().float()
    return {
        "different_values": int(delta.count_nonzero()),
        "maximum_absolute_difference": float(delta.abs().max()),
    }


def replay_expert_projections(
    checkpoint: Path, first_layer: dict[str, torch.Tensor], token_index: int
) -> None:
    gate, up, down, shared_gate, router = read_weights(
        checkpoint,
        (
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
            "mlp.shared_expert_gate",
            "mlp.gate",
        ),
    )
    gate_up_weight = torch.cat((gate, up))
    source = first_layer["mlp_normalized"].flatten().cuda()
    native_shared = first_layer["shared_before_gate"]
    native_gate = first_layer["shared_gate"]
    native_router = first_layer["router_logits"]
    native_expert_ids = first_layer["routed_expert_ids"].flatten().cuda()
    for token_count in TOKEN_COUNTS:
        inputs = source.new_zeros((token_count, source.numel()))
        inputs[token_index] = source
        packed_projection = functional.linear(inputs, gate_up_weight)
        activation = silu_and_mul(packed_projection)
        shared = functional.linear(activation, down)[token_index]
        gate_value = functional.linear(inputs, shared_gate)[token_index]
        router_logits = functional.linear(inputs, router)[token_index]
        top_ids = router_logits.topk(native_expert_ids.numel()).indices
        print(
            json.dumps(
                {
                    "expert_projection_token_count": token_count,
                    "shared_before_gate": report_difference(shared, native_shared),
                    "shared_gate": report_difference(gate_value, native_gate),
                    "shared_after_gate": report_difference(
                        shared * gate_value.sigmoid(), first_layer["shared_output"]
                    ),
                    "router_logits": report_difference(router_logits, native_router),
                    "same_selected_experts": torch.equal(
                        top_ids.sort().values, native_expert_ids.sort().values
                    ),
                }
            ),
            flush=True,
        )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--layer", type=Path, required=True)
    parser.add_argument("--token-index", type=int, required=True)
    arguments = parser.parse_args()
    prefix = torch.load(arguments.prefix, weights_only=True)
    first_layer = torch.load(arguments.layer, weights_only=True)[0]
    keys = prefix["key_rows"][0].cuda()
    values = prefix["value_rows"][0].cuda()
    target_length, key_head_count, head_dimension = keys.shape
    query = first_layer["query_rope"].view(-1, head_dimension).cuda()
    native_context = first_layer["attention_context"].view_as(query).cuda()
    native_output = first_layer["attention_output"].cuda()
    (output_weight,) = read_weights(arguments.checkpoint, ("self_attn.o_proj",))
    # note (ratish): the original logs expose total rows, not individual peer lengths.
    layouts = (
        (target_length,),
        (target_length, 296 - target_length),
        (146, target_length, 121, 136, 171, 118, 112, 138, 140, 178),
        (target_length, *([143] * 10), 1723 - target_length - 1430),
    )
    for request_lengths in layouts:
        target_rank = 1 if len(request_lengths) == 10 else 0
        starts = (0, *accumulate(request_lengths))
        target_start = starts[target_rank]
        target_end = starts[target_rank + 1]
        token_count = starts[-1]
        absolute_token_index = target_start + arguments.token_index
        packed_keys = keys.new_zeros((token_count, 1, key_head_count, head_dimension))
        packed_values = torch.ones_like(packed_keys)
        packed_keys[target_start:target_end, 0] = keys
        packed_values[target_start:target_end, 0] = values
        packed_queries = query.new_zeros((token_count, *query.shape))
        packed_queries[absolute_token_index] = query
        request_lengths_device = torch.tensor(
            request_lengths, dtype=torch.int32, device=query.device
        )
        starts_device = torch.tensor(starts, dtype=torch.int32, device=query.device)
        page_table = torch.zeros(
            len(request_lengths),
            max(request_lengths),
            dtype=torch.int32,
            device=query.device,
        )
        for request_index, request_length in enumerate(request_lengths):
            page_table[request_index, :request_length] = torch.arange(
                starts[request_index],
                starts[request_index + 1],
                dtype=torch.int32,
                device=query.device,
            )
        contexts = []
        for peer_value in (1, -1):
            packed_values[:target_start].fill_(peer_value)
            packed_values[target_end:].fill_(peer_value)
            context = flash_attn_with_kvcache(
                q=packed_queries,
                k_cache=packed_keys,
                v_cache=packed_values,
                page_table=page_table,
                cache_seqlens=request_lengths_device,
                cu_seqlens_q=starts_device,
                cu_seqlens_k_new=starts_device,
                max_seqlen_q=max(request_lengths),
                causal=True,
                softmax_scale=head_dimension**-0.5,
                num_splits=0,
                ver=3,
            )
            contexts.append(context[absolute_token_index].clone())
        print(
            json.dumps(
                {
                    "request_lengths": request_lengths,
                    "raw_token_count": token_count,
                    "same_as_native_attention": torch.equal(
                        contexts[0], native_context
                    ),
                    "other_request_values_do_not_leak": torch.equal(*contexts),
                    "native_context_max_abs_difference": float(
                        (contexts[0].float() - native_context.float()).abs().max()
                    ),
                }
            ),
            flush=True,
        )
    for token_count in TOKEN_COUNTS:
        contexts = native_context.new_zeros((token_count, native_context.numel()))
        contexts[arguments.token_index] = native_context.flatten()
        projected = functional.linear(contexts, output_weight)[arguments.token_index]
        print(
            json.dumps(
                {
                    "output_projection_token_count": token_count,
                    "same_as_native_attention_projection": torch.equal(
                        projected, native_output
                    ),
                    "native_projection_max_abs_difference": float(
                        (projected.float() - native_output.float()).abs().max()
                    ),
                }
            ),
            flush=True,
        )
    replay_expert_projections(arguments.checkpoint, first_layer, arguments.token_index)


if __name__ == "__main__":
    main()
