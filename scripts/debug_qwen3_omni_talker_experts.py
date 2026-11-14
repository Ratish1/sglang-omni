"""Isolate native routed experts in otherwise checkpoint Talker recurrence."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.layers.activation import SiluAndMul
from torch.utils.hooks import RemovableHandle
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeTalkerTextExperts,
    Qwen3OmniMoeTalkerTextMLP,
    Qwen3OmniMoeTalkerTextTopKRouter,
)

from scripts.qwen3_omni_talker_expert_rounding import expert_rounding_stages
from scripts.qwen3_omni_talker_expert_runtime import (
    NativeRuntime,
    ObservedExpertProfiles,
    combine_expert_contributions,
    load_native_expert_weights,
    native_expert_contributions,
)


@dataclass(kw_only=True)
class NativeTalkerExperts:
    reference: Qwen3OmniMoeTalkerForConditionalGeneration
    capture_directory: Path
    route_precision: Literal["checkpoint", "float32"]
    shared_activation_semantics: Literal["native", "checkpoint"]
    output_arithmetic: Literal[
        "native",
        "native-reduction",
        "checkpoint",
        "rounded-contributions",
        "bf16-reduction",
        "checkpoint-reduction",
        "fused-activation-checkpoint-reduction",
    ] = "native"
    native_output_arithmetic: Literal[
        "native", "rounded-contributions", "checkpoint-reduction"
    ] = "native"
    enabled: bool = False
    calls: int = 0
    router_calls: int = 0
    kernel_calls: int = 0
    routing_semantics: Literal[
        "reconstructed", "indices", "native", "rounded", "indices-rounded"
    ] = "reconstructed"
    shared_enabled: bool = False
    shared_calls: int = 0
    is_shared_gate_batch_replay_enabled: bool = False
    shared_gate_decode_rows: int | None = None
    shared_gate_calls: int = 0
    shared_gate_exact_input_calls: int = 0
    shared_gate_snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)
    profiles: ObservedExpertProfiles | None = None
    gemm1_profile: int | None = None
    used_profiles: set[tuple[int, int]] = field(default_factory=set)
    runtime: NativeRuntime = field(init=False)
    weights: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    shared_weights: list[torch.Tensor] = field(default_factory=list)
    shared_activation: SiluAndMul = field(default_factory=SiluAndMul)
    handles: list[RemovableHandle] = field(default_factory=list)

    def install(self) -> None:
        self.runtime = NativeRuntime.model_validate_json(
            (self.capture_directory / "moe-runtime.json").read_bytes()
        )
        self.weights = load_native_expert_weights(self.reference, self.runtime)
        for layer_index, layer in enumerate(self.reference.model.layers):
            if self.shared_enabled:
                self.shared_weights.append(
                    torch.cat(
                        (
                            layer.mlp.shared_expert.gate_proj.weight,
                            layer.mlp.shared_expert.up_proj.weight,
                        )
                    ).contiguous()
                )
                self.handles.append(
                    layer.mlp.shared_expert.register_forward_hook(
                        partial(self.shared_expert, layer_index=layer_index)
                    )
                )
            else:
                pass
            if self.is_shared_gate_batch_replay_enabled:
                self.handles.append(
                    layer.mlp.shared_expert_gate.register_forward_hook(
                        partial(self.shared_gate, layer_index=layer_index)
                    )
                )
            else:
                pass
            self.handles.extend(
                [
                    layer.mlp.gate.register_forward_hook(self.routing),
                    layer.mlp.experts.register_forward_hook(
                        partial(self.experts, layer_index=layer_index)
                    ),
                ]
            )

    def shared_gate(
        self,
        module: torch.nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        assert self.profiles is not None
        layer_count = len(self.reference.model.layers)
        assert layer_index == self.shared_gate_calls % layer_count
        frame = self.shared_gate_calls // layer_count
        hidden_states = inputs[0].reshape(-1, inputs[0].shape[-1])
        if frame > 0 and self.shared_gate_decode_rows is not None:
            assert len(hidden_states) == 1
            batch_hidden_states = hidden_states.repeat(self.shared_gate_decode_rows, 1)
            start, stop = 0, 1
        else:
            if layer_index == 0:
                self.shared_gate_snapshots = torch.load(
                    self.profiles.source_capture / f"{frame:04d}-operators.pt",
                    weights_only=True,
                )
            else:
                pass
            snapshot = self.shared_gate_snapshots[layer_index]
            batch_hidden_states = snapshot["batch_mlp_normalized"].to(
                hidden_states.device
            )
            if frame == 0:
                start, stop = snapshot["prefix_token_rows"].tolist()
            else:
                start = int(snapshot["batch_selected_row"])
                stop = start + 1
            self.shared_gate_exact_input_calls += torch.equal(
                batch_hidden_states[start:stop], hidden_states
            )
        assert batch_hidden_states[start:stop].shape == hidden_states.shape
        assert module.bias is None
        assert batch_hidden_states.dtype == hidden_states.dtype == module.weight.dtype
        batch_hidden_states[start:stop] = hidden_states
        gate = torch.nn.functional.linear(batch_hidden_states, module.weight)
        self.shared_gate_calls += 1
        return gate[start:stop].view_as(output)

    def shared_expert(
        self,
        module: Qwen3OmniMoeTalkerTextMLP,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        assert layer_index == self.shared_calls % len(self.shared_weights)
        hidden_states = inputs[0].reshape(-1, inputs[0].shape[-1])
        assert hidden_states.dtype == module.down_proj.weight.dtype == torch.bfloat16
        gate_up = torch.nn.functional.linear(
            hidden_states, self.shared_weights[layer_index]
        )
        if self.shared_activation_semantics == "checkpoint":
            activated = self.shared_activation.forward_native(gate_up)
        else:
            activated = self.shared_activation(gate_up)
        self.shared_calls += 1
        return torch.nn.functional.linear(activated, module.down_proj.weight).view_as(
            output
        )

    def routing(
        self,
        module: Qwen3OmniMoeTalkerTextTopKRouter,
        inputs: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enabled and self.route_precision == "float32":
            logits, _, indices = output
            if self.routing_semantics != "reconstructed":
                weights, indices = moe_fused_gate(
                    logits,
                    None,
                    module.top_k,
                    scoring_func="softmax",
                    renormalize=module.norm_topk_prob,
                )
                indices = indices.long()
            else:
                pass
            if self.routing_semantics in (
                "reconstructed",
                "indices",
                "indices-rounded",
            ):
                probabilities = logits.float().softmax(dim=-1)
                weights = probabilities.gather(1, indices)
                if module.norm_topk_prob:
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                else:
                    pass
            else:
                pass
            if self.routing_semantics in ("rounded", "indices-rounded"):
                weights = weights.to(logits.dtype).float()
            else:
                pass
            self.router_calls += 1
            return logits, weights, indices
        else:
            return output

    def experts(
        self,
        module: Qwen3OmniMoeTalkerTextExperts,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        else:
            pass
        hidden_states, indices, weights = inputs
        assert hidden_states.dtype == torch.bfloat16
        assert indices.shape == weights.shape
        assert weights.dtype == (
            torch.float32 if self.route_precision == "float32" else torch.bfloat16
        )
        packed, down = self.weights[layer_index]
        assert layer_index == self.calls % len(self.weights)
        if self.output_arithmetic == "checkpoint":
            self.calls += 1
            return output
        elif self.output_arithmetic == "fused-activation-checkpoint-reduction":
            assert torch.equal(weights, weights.bfloat16().float())
            stages = expert_rounding_stages(module, hidden_states, indices, weights)
            assert torch.equal(
                stages["checkpoint_expert_order_bfloat16_sum"]
                .contiguous()
                .view(torch.uint8),
                output.contiguous().view(torch.uint8),
            ), (layer_index, self.calls, "checkpoint expert baseline")
            self.calls += 1
            return stages["fused_activation_checkpoint_sum"]
        else:
            pass
        if self.profiles is not None:
            profiles = (
                self.profiles.prefill[layer_index]
                if self.calls < len(self.weights)
                else self.profiles.decode[layer_index]
            )
        else:
            profiles = (-1, -1)
        if self.gemm1_profile is not None:
            profiles = (self.gemm1_profile, profiles[1])
        else:
            pass
        self.used_profiles.add(profiles)
        replay_inputs = hidden_states.contiguous().clone()
        fused_output = cutlass_fused_moe(
            input=replay_inputs,
            token_selected_experts=indices.to(torch.int32).contiguous(),
            token_final_scales=weights.float().contiguous(),
            fc1_expert_weights=packed,
            fc2_expert_weights=down,
            output_dtype=torch.bfloat16,
            quant_scales=None,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=self.runtime.tune_max_num_tokens,
            use_fused_finalize=self.runtime.use_fused_finalize,
            profile_ids=list(profiles),
        )[0]
        assert torch.equal(replay_inputs, hidden_states)
        assert fused_output.shape == output.shape
        if self.output_arithmetic in (
            "native-reduction",
            "rounded-contributions",
            "bf16-reduction",
            "checkpoint-reduction",
        ):
            contributions = native_expert_contributions(
                hidden_states, indices, packed, down, self.runtime, profiles
            )
            reproduced = combine_expert_contributions(
                contributions, indices, weights, self.native_output_arithmetic
            )
            assert torch.equal(
                reproduced.contiguous().view(torch.uint8),
                fused_output.contiguous().view(torch.uint8),
            ), (layer_index, self.calls)
            fused_output = combine_expert_contributions(
                contributions,
                indices,
                weights,
                (
                    "native"
                    if self.output_arithmetic == "native-reduction"
                    else self.output_arithmetic
                ),
            )
            self.kernel_calls += len(contributions)
        else:
            pass
        self.calls += 1
        self.kernel_calls += 1
        return fused_output
