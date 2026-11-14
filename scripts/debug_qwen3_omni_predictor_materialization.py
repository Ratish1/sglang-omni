"""Audit fused predictor operators without changing their recurrent outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from io import TextIOWrapper
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.debug_qwen3_omni_predictor_cache import NativePredictorReplay
from sglang_omni.utils.predictor_layers import (
    FusedPredictorLayers,
    PredictorLayerShape,
    down_add,
    o_proj_add,
)


class ObservedPredictorLayers(FusedPredictorLayers):
    def __init__(
        self,
        shape: PredictorLayerShape,
        device: torch.device,
        destination: TextIOWrapper,
    ) -> None:
        super().__init__(shape, 2, device, torch.bfloat16)
        self.destination: TextIOWrapper = destination
        self.request_id: str = ""
        self.frame: int = 0
        self.operator_calls: int = 0

    def forward(
        self,
        *,
        layers: nn.ModuleList,
        final_norm: nn.Module,
        token_embeds: torch.Tensor,
        batch_size: int,
        cache_len: int,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        assert batch_size == 1
        tokens = token_embeds.shape[1]
        end = cache_len + tokens
        assert torch.equal(
            positions, torch.arange(cache_len, end, device=positions.device)
        )
        hidden = token_embeds
        for layer_index, layer in enumerate(layers):
            inputs = hidden.reshape(tokens, self.shape.hidden_size).clone()
            hidden = super().forward(
                layers=nn.ModuleList([layer]),
                final_norm=nn.Identity(),
                token_embeds=hidden,
                batch_size=batch_size,
                cache_len=cache_len,
                positions=positions,
                k_cache=k_cache[layer_index : layer_index + 1],
                v_cache=v_cache[layer_index : layer_index + 1],
            )
            actual_output = hidden.reshape_as(inputs).clone()
            actual_values = v_cache[layer_index, 0, cache_len:end].reshape(tokens, -1)
            query_width = self.shape.num_q_heads * self.shape.head_dim
            value_start = query_width + self.shape.num_kv_heads * self.shape.head_dim
            projection = layer.self_attn.qkv_proj.weight[value_start:]
            norm = layer.input_layernorm
            normalized = norm(inputs)
            checkpoint_values = F.linear(normalized, projection)
            inverse_rms = torch.rsqrt(
                inputs.double().square().mean(-1, keepdim=True) + norm.variance_epsilon
            )
            mathematical_values = F.linear(
                inputs.double() * inverse_rms * norm.weight.double(),
                projection.double(),
            ).to(torch.bfloat16)
            preweighted = (inputs.float() * norm.weight.float()).to(torch.bfloat16)
            folded_values = (
                F.linear(preweighted.float(), projection.float()) * inverse_rms.float()
            ).to(torch.bfloat16)

            query = (
                self.q[:tokens]
                .view(1, tokens, self.shape.num_q_heads, self.shape.head_dim)
                .transpose(1, 2)
            )
            keys = k_cache[layer_index, :, :end].transpose(1, 2)
            values = v_cache[layer_index, :, :end].transpose(1, 2)
            heads_per_group = self.shape.num_q_heads // self.shape.num_kv_heads
            keys = keys.repeat_interleave(heads_per_group, dim=1)
            values = values.repeat_interleave(heads_per_group, dim=1)
            visible = (
                torch.arange(end, device=positions.device)[None, :]
                <= positions[:, None]
            )
            scores = (
                query.double() @ keys.double().transpose(-1, -2)
            ) * self.shape.head_dim**-0.5
            scores = scores.masked_fill(~visible, float("-inf"))
            mathematical_attention = (
                (scores.softmax(-1) @ values.double())
                .transpose(1, 2)
                .reshape(tokens, -1)
                .to(torch.bfloat16)
            )
            scores = (
                query.float() @ keys.float().transpose(-1, -2)
            ) * self.shape.head_dim**-0.5
            scores = scores.masked_fill(~visible, float("-inf"))
            weights = (scores - scores.amax(-1, keepdim=True)).exp()
            folded_attention = (
                (
                    (weights.to(torch.bfloat16).float() @ values.float())
                    / weights.sum(-1, keepdim=True)
                )
                .transpose(1, 2)
                .reshape(tokens, -1)
                .to(torch.bfloat16)
            )
            actual_attention = self.attention_output[:tokens]

            after_attention = torch.empty_like(inputs)
            o_proj_add(
                attention_output=actual_attention.view(
                    1, tokens, self.shape.num_q_heads, self.shape.head_dim
                ).transpose(1, 2),
                tokens_per_row=tokens,
                weight=layer.self_attn.o_proj.weight,
                residual_in=inputs,
                residual_out=after_attention,
                partials=self.partials,
                counters=self.counters,
                shape=self.shape,
                fitted_stages=self.fitted_stages,
            )
            norm = layer.post_attention_layernorm
            projection = layer.mlp.gate_up_proj.weight
            normalized = norm(after_attention)
            gate, up = F.linear(normalized, projection).chunk(2, dim=-1)
            checkpoint_activation = (F.silu(gate.float()) * up.float()).to(
                torch.bfloat16
            )
            inverse_rms = torch.rsqrt(
                after_attention.double().square().mean(-1, keepdim=True)
                + norm.variance_epsilon
            )
            preweighted = (after_attention.float() * norm.weight.float()).to(
                torch.bfloat16
            )
            products = (
                F.linear(preweighted.float(), projection.float()) * inverse_rms.float()
            ).to(torch.bfloat16)
            gate, up = products.float().chunk(2, dim=-1)
            folded_activation = (F.silu(gate) * up).to(torch.bfloat16)
            gate, up = (
                F.linear(
                    after_attention.double() * inverse_rms * norm.weight.double(),
                    projection.double(),
                )
                .to(torch.bfloat16)
                .float()
                .chunk(2, dim=-1)
            )
            mathematical_activation = (F.silu(gate) * up).to(torch.bfloat16)
            actual_activation = self.activated[:tokens]
            down_add(
                activated=actual_activation,
                weight=layer.mlp.down_proj.weight,
                residual=after_attention,
                partials=self.partials,
                counters=self.counters,
                shape=self.shape,
                fitted_stages=self.fitted_stages,
            )
            assert torch.equal(
                after_attention.view(torch.uint8), actual_output.view(torch.uint8)
            )
            assert not bool(self.counters.count_nonzero())
            record = {
                "request_id": self.request_id,
                "frame": self.frame,
                "predictor_step": max(cache_len - 1, 0),
                "layer": layer_index,
                "tokens": tokens,
                "native_residual_closure_bytes_exact": True,
                "values_checkpoint_materialization": asdict(
                    compare_tensors(actual_values, checkpoint_values)
                ),
                "values_folded_materialization": asdict(
                    compare_tensors(actual_values, folded_values)
                ),
                "values_float64_formula": asdict(
                    compare_tensors(actual_values, mathematical_values)
                ),
                "attention_float64_formula": asdict(
                    compare_tensors(actual_attention, mathematical_attention)
                ),
                "attention_unnormalized_bf16_weights": asdict(
                    compare_tensors(actual_attention, folded_attention)
                ),
                "activation_checkpoint_norm": asdict(
                    compare_tensors(actual_activation, checkpoint_activation)
                ),
                "activation_folded_norm": asdict(
                    compare_tensors(actual_activation, folded_activation)
                ),
                "activation_float64_norm_projection": asdict(
                    compare_tensors(actual_activation, mathematical_activation)
                ),
            }
            self.destination.write(json.dumps(record) + "\n")
            self.operator_calls += 1
        return final_norm(hidden.reshape(-1, self.shape.hidden_size)).reshape_as(hidden)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"]
    reference.eval()
    predictor = NativePredictorReplay(
        reference=reference,
        capture_directory=arguments.capture_directory,
        output_directory=arguments.output_directory,
        model_path=arguments.model_path,
        model_revision=arguments.model_revision,
        feedback_reduction="checkpoint",
    )
    ordinary = predictor.fused
    reports = []
    with (arguments.output_directory / "operators.jsonl").open("x") as destination:
        observed_layers = ObservedPredictorLayers(
            ordinary.shape, reference.device, destination
        )
        for request_id in arguments.request_id:
            directory = arguments.capture_directory / request_id
            finished = torch.load(directory / "finished.pt", weights_only=True)
            files = sorted(directory.glob("*-output.pt"))
            assert [p.name for p in files] == [
                f"{frame:04d}-output.pt" for frame in range(len(finished["output_ids"]))
            ]
            checked = 0
            for frame, filename in enumerate(files):
                output = torch.load(filename, weights_only=True)
                assert int(output["codec_ids"][0]) == int(finished["output_ids"][frame])
                if frame == len(files) - 1:
                    assert int(output["codec_ids"][0]) == config.codec_eos_token_id
                else:
                    main_code = output["codec_ids"][:1].to(reference.device).long()
                    query = output["hidden_states"].view(1, 1, -1).to(reference.device)
                    predictor.fused = ordinary
                    ordinary_codes, ordinary_feedback = predictor.generate(
                        main_code, query
                    )
                    assert torch.equal(
                        ordinary_codes[0].cpu(), output["codec_ids"].long()
                    )
                    assert torch.equal(
                        ordinary_feedback[0, 0].cpu().contiguous().view(torch.uint8),
                        output["codec_feedback"].contiguous().view(torch.uint8),
                    )
                    observed_layers.request_id = request_id
                    observed_layers.frame = frame
                    predictor.fused = observed_layers
                    observed_codes, observed_feedback = predictor.generate(
                        main_code, query
                    )
                    assert torch.equal(observed_codes, ordinary_codes)
                    assert torch.equal(
                        observed_feedback.view(torch.uint8),
                        ordinary_feedback.view(torch.uint8),
                    )
                    checked += 1
                    destination.flush()
            reports.append(
                {
                    "request_id": request_id,
                    "consumed_frames": checked,
                    "native_codes_exact": True,
                    "native_feedback_bytes_exact": True,
                }
            )
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(
                    {
                        "requests": reports,
                        "operator_calls": observed_layers.operator_calls,
                        "scope": "Actual native predictor inputs and free residual recurrence repeated without changing outputs. Every five-layer operator boundary, every consumed residual prefix and all early/middle/late frames. Extra independent calculations never feed the native recurrence. Float64 and materialization formulas are comparisons, not bitwise GEMM/rsqrt emulators or numerical-gold speech claims. No audio, tensor dumps, model defaults or native kernels changed.",
                    },
                    indent=2,
                )
            )
            print(json.dumps(reports[-1]), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
