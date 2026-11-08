"""Profile and decompose the actual predictor pass at a saved codebook choice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.debug_qwen3_omni_operator_audit import compare_tensors
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.stages import create_talker_ar_executor_from_config
from sglang_omni.utils.predictor_layers import (
    attention_inputs,
    decode_attention,
    down_add,
    mlp_up,
    o_proj_add,
)


def predictor_prefix(
    native: Qwen3OmniTalker,
    codes: torch.Tensor,
    hidden_row: torch.Tensor,
    target_group: int,
) -> tuple[torch.Tensor, int]:
    embeddings = torch.cat(
        (hidden_row, native.model.codec_embedding(codes[:, :1])), dim=1
    )
    cache_length = 0
    for group in range(1, target_group):
        native.predictor_forward_tokens(
            token_embeds=embeddings, batch_size=1, cache_len=cache_length
        )
        cache_length += embeddings.shape[1]
        embeddings = native.code_predictor.model.codec_embedding[group - 1](
            codes[:, group : group + 1]
        )
    return embeddings, cache_length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--predictor-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.set_float32_matmul_precision("highest")
    scheduler = create_talker_ar_executor_from_config(
        arguments.model_path,
        gpu_id=0,
        max_running_requests=arguments.max_running_requests,
        total_gpu_memory_fraction=arguments.memory_fraction,
        code2wav_in_process=True,
    )
    native = scheduler.model_runner.model
    fused = native.predictor_fused_layers
    assert fused is not None
    shape = fused.shape
    directory = arguments.capture_directory / "talker_state"
    conditioning = torch.load(
        directory / f"{arguments.request_id}.conditioning.pt",
        map_location="cuda",
        weights_only=True,
    )
    captured_codes = torch.load(
        directory / f"{arguments.request_id}.pt", map_location="cuda", weights_only=True
    ).long()
    captured_predictor = torch.load(
        arguments.predictor_directory / f"{arguments.request_id}.{arguments.frame}.pt",
        map_location="cuda",
        weights_only=True,
    )
    assert 1 <= arguments.group < native.config.num_code_groups
    codes = captured_codes[arguments.frame : arguments.frame + 1]
    hidden_row = conditioning["native_backbone_hidden_states"][arguments.frame].view(
        1, 1, -1
    )
    assert torch.equal(hidden_row.flatten(), captured_predictor["hidden_states"])
    reports = []
    with torch.inference_mode():
        token_embeddings, cache_length = predictor_prefix(
            native, codes, hidden_row, arguments.group
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profile:
            expected_hidden = native.predictor_forward_tokens(
                token_embeds=token_embeddings, batch_size=1, cache_len=cache_length
            ).clone()
            expected_logits = native.code_predictor.lm_head[arguments.group - 1](
                expected_hidden[:, -1:]
            )[0].clone()
        profile.export_chrome_trace(
            str(arguments.output_directory / "native-pass.trace.json")
        )
        assert torch.equal(
            expected_logits.float(),
            captured_predictor["native_logits"][
                :, arguments.group - 1 : arguments.group
            ],
        )
        prefix_keys = native.predictor_k_cache[:, :1, :cache_length].clone()
        prefix_values = native.predictor_v_cache[:, :1, :cache_length].clone()
        head = native.code_predictor.lm_head[arguments.group - 1]
        history_controls = {"fused_past_fused_current": expected_logits}
        native.predictor_fused_layers = None
        plain_current = native.predictor_forward_tokens(
            token_embeds=token_embeddings, batch_size=1, cache_len=cache_length
        )
        history_controls["fused_past_plain_current"] = head(plain_current[:, -1:])[
            0
        ].clone()
        rebuilt_embeddings, rebuilt_cache_length = predictor_prefix(
            native, codes, hidden_row, arguments.group
        )
        assert (
            torch.equal(token_embeddings, rebuilt_embeddings)
            and cache_length == rebuilt_cache_length
        )
        native.predictor_fused_layers = fused
        fused_current = native.predictor_forward_tokens(
            token_embeds=token_embeddings, batch_size=1, cache_len=cache_length
        )
        history_controls["plain_past_fused_current"] = head(fused_current[:, -1:])[
            0
        ].clone()
        native.predictor_fused_layers = None
        plain_current = native.predictor_forward_tokens(
            token_embeds=token_embeddings, batch_size=1, cache_len=cache_length
        )
        history_controls["plain_past_plain_current"] = head(plain_current[:, -1:])[
            0
        ].clone()
        assert torch.equal(
            history_controls["plain_past_plain_current"].float(),
            captured_predictor["plain_logits"][
                :, arguments.group - 1 : arguments.group
            ],
        )
        native.predictor_fused_layers = fused
        native.predictor_k_cache[:, :1, :cache_length].copy_(prefix_keys)
        native.predictor_v_cache[:, :1, :cache_length].copy_(prefix_values)
        torch.save(
            {name: logits.cpu() for name, logits in history_controls.items()},
            arguments.output_directory / "history-controls.pt",
        )
        token_count = token_embeddings.shape[1]
        hidden_states = token_embeddings.reshape(token_count, -1)
        residual = fused.residual[:token_count]
        positions = native.predictor_positions[
            cache_length : cache_length + token_count
        ]
        tensor_captures = {}
        for layer_index, layer in enumerate(native.code_predictor.model.layers):
            attention = layer.self_attn
            layer_keys = native.predictor_k_cache[layer_index, :1]
            layer_values = native.predictor_v_cache[layer_index, :1]
            input_rows = hidden_states.clone()
            attention_inputs(
                x=hidden_states,
                tokens_per_row=token_count,
                norm=layer.input_layernorm,
                attention=attention,
                q_out=fused.q[:token_count],
                positions=positions,
                k_cache=layer_keys,
                v_cache=layer_values,
                partials=fused.partials,
                sum_sq_partials=fused.sum_sq_partials,
                counters=fused.counters,
                shape=shape,
                fitted_stages=fused.fitted_stages,
            )
            queries = fused.q[:token_count].clone()
            end = cache_length + token_count
            current_keys = layer_keys[:, :end].clone()
            current_values = layer_values[:, :end].clone()
            decode_attention(
                q=fused.q[:token_count],
                tokens_per_row=token_count,
                k_cache=layer_keys,
                v_cache=layer_values,
                end=end,
                attention_output=fused.attention_output[:token_count],
            )
            attended = fused.attention_output[:token_count].clone()
            query_heads = queries.view(
                token_count, shape.num_q_heads, shape.head_dim
            ).float()
            key_heads = (
                current_keys[0]
                .float()
                .repeat_interleave(shape.num_q_heads // shape.num_kv_heads, dim=1)
            )
            value_heads = (
                current_values[0]
                .float()
                .repeat_interleave(shape.num_q_heads // shape.num_kv_heads, dim=1)
            )
            scores = (
                torch.einsum("thd,shd->ths", query_heads, key_heads)
                * shape.head_dim**-0.5
            )
            visible = (
                torch.arange(end, device=queries.device)[None] <= positions[:, None]
            )
            scores.masked_fill_(~visible[:, None], -torch.inf)
            expected_attention = torch.einsum(
                "ths,shd->thd", scores.softmax(-1), value_heads
            ).reshape_as(attended)
            normalized_input = F.rms_norm(
                input_rows.float(),
                (shape.hidden_size,),
                layer.input_layernorm.weight.float(),
                layer.input_layernorm.variance_epsilon,
            )
            projected = F.linear(normalized_input, attention.qkv_proj.weight.float())
            query_projection, key_projection, value_projection = projected.split(
                (
                    shape.num_q_heads * shape.head_dim,
                    shape.num_kv_heads * shape.head_dim,
                    shape.num_kv_heads * shape.head_dim,
                ),
                dim=-1,
            )
            float_rotated = []
            cosine_sine = attention.rotary_emb.cos_sin_cache[positions].float()
            cosine, sine = cosine_sine.chunk(2, dim=-1)
            for vectors, norm in (
                (query_projection, attention.q_norm),
                (key_projection, attention.k_norm),
            ):
                vectors = F.rms_norm(
                    vectors.view(token_count, -1, shape.head_dim),
                    (shape.head_dim,),
                    norm.weight.float(),
                    norm.variance_epsilon,
                )
                first_half, second_half = vectors.chunk(2, dim=-1)
                float_rotated.append(
                    torch.cat(
                        (
                            first_half * cosine[:, None] - second_half * sine[:, None],
                            second_half * cosine[:, None] + first_half * sine[:, None],
                        ),
                        dim=-1,
                    )
                )
            o_proj_add(
                attention_output=fused.attention_output[:token_count]
                .view(1, token_count, shape.num_q_heads, shape.head_dim)
                .transpose(1, 2),
                tokens_per_row=token_count,
                weight=attention.o_proj.weight,
                residual_in=hidden_states,
                residual_out=residual,
                partials=fused.partials,
                counters=fused.counters,
                shape=shape,
                fitted_stages=fused.fitted_stages,
            )
            attention_residual = residual.clone()
            expected_attention_residual = (
                F.linear(attended.float(), attention.o_proj.weight.float()).to(
                    residual.dtype
                )
                + input_rows
            )
            hidden_states = residual
            mlp_up(
                residual=residual,
                norm=layer.post_attention_layernorm,
                weight=layer.mlp.gate_up_proj.weight,
                activated=fused.activated[:token_count],
                shape=shape,
                fitted_stages=fused.fitted_stages,
            )
            activated = fused.activated[:token_count].clone()
            normalized_mlp = F.rms_norm(
                attention_residual.float(),
                (shape.hidden_size,),
                layer.post_attention_layernorm.weight.float(),
                layer.post_attention_layernorm.variance_epsilon,
            )
            gate, up = F.linear(
                normalized_mlp, layer.mlp.gate_up_proj.weight.float()
            ).chunk(2, dim=-1)
            expected_activated = (
                F.silu(gate.to(activated.dtype).float())
                * up.to(activated.dtype).float()
            )
            down_add(
                activated=fused.activated[:token_count],
                weight=layer.mlp.down_proj.weight,
                residual=residual,
                partials=fused.partials,
                counters=fused.counters,
                shape=shape,
                fitted_stages=fused.fitted_stages,
            )
            expected_mlp_residual = (
                F.linear(activated.float(), layer.mlp.down_proj.weight.float()).to(
                    residual.dtype
                )
                + attention_residual
            )
            assert not fused.counters.any()
            comparisons = {
                "query_vs_float32": compare_tensors(
                    queries.view_as(float_rotated[0]), float_rotated[0]
                ),
                "current_key_vs_float32": compare_tensors(
                    current_keys[0, cache_length:end], float_rotated[1]
                ),
                "current_value_vs_float32": compare_tensors(
                    current_values[0, cache_length:end],
                    value_projection.view(
                        token_count, shape.num_kv_heads, shape.head_dim
                    ),
                ),
                "attention_vs_float32": compare_tensors(attended, expected_attention),
                "attention_residual_vs_float32_projection": compare_tensors(
                    attention_residual, expected_attention_residual
                ),
                "mlp_activation_vs_float32_projection": compare_tensors(
                    activated, expected_activated
                ),
                "mlp_residual_vs_float32_projection": compare_tensors(
                    residual, expected_mlp_residual
                ),
            }
            reports.append({"layer": layer_index, "comparisons": comparisons})
            tensor_captures[f"layer_{layer_index}"] = {
                "input": input_rows.cpu(),
                "queries": queries.cpu(),
                "keys": current_keys.cpu(),
                "values": current_values.cpu(),
                "attention_output": attended.cpu(),
                "attention_residual": attention_residual.cpu(),
                "mlp_activation": activated.cpu(),
                "mlp_residual": residual.cpu(),
            }
        actual_hidden = native.code_predictor.model.norm(residual).view(
            1, token_count, -1
        )
        actual_logits = native.code_predictor.lm_head[arguments.group - 1](
            actual_hidden[:, -1:]
        )[0]
        assert torch.equal(actual_hidden, expected_hidden)
        assert torch.equal(actual_logits, expected_logits)
        assert torch.equal(prefix_keys, native.predictor_k_cache[:, :1, :cache_length])
        assert torch.equal(
            prefix_values, native.predictor_v_cache[:, :1, :cache_length]
        )
        torch.save(tensor_captures, arguments.output_directory / "boundaries.pt")
        result = {
            "request_id": arguments.request_id,
            "frame": arguments.frame,
            "group": arguments.group,
            "recorded_logits_exact": True,
            "decomposed_hidden_exact": True,
            "decomposed_logits_exact": True,
            "prefix_unchanged": True,
            "history_controls": {
                name: {
                    "argmax": int(logits.argmax(-1)),
                    "top_two_gap": float(
                        logits.float()
                        .topk(2, dim=-1)
                        .values.diff(dim=-1)
                        .neg()
                        .flatten()[0]
                    ),
                }
                for name, logits in history_controls.items()
            },
            "layers": reports,
        }
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(result, indent=2)
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
