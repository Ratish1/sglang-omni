# SPDX-License-Identifier: Apache-2.0
"""Copy native decode boundaries without replacing model operands."""

from typing import Protocol

import torch
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from sglang_omni.vendor.sglang.layers import residual_batch


class OutputRecorder(Protocol):
    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    ) -> None: ...


class NativeLayerProbe:
    def __init__(
        self, layer: nn.Module, predictor_heads: nn.ModuleList, layer_index: int
    ) -> None:
        self.layer_index = layer_index
        self.active: dict[str, torch.Tensor] | None = None
        self.decode_snapshot: dict[str, torch.Tensor] | None = None
        self.predictor_groups = len(predictor_heads)
        self.snapshots: list[dict[str, torch.Tensor]] = []
        self.eager: dict[str, torch.Tensor] = {}
        layer.register_forward_pre_hook(self.begin, with_kwargs=True)
        layer.self_attn.register_forward_pre_hook(
            self.attention_input, with_kwargs=True
        )
        layer.self_attn.qkv_proj.register_forward_hook(self.output("qkv_projection"))
        layer.self_attn.rotary_emb.register_forward_pre_hook(self.before_rope)
        layer.self_attn.attn.register_forward_pre_hook(self.qkv_after_rope)
        layer.self_attn.attn.register_forward_hook(self.output("attention_context"))
        layer.self_attn.register_forward_hook(self.output("attention_projection"))
        layer.mlp.register_forward_pre_hook(self.feed_forward_input)
        layer.mlp.gate.register_forward_hook(self.output("router_logits"))
        layer.mlp.topk.register_forward_hook(self.router_selection)
        layer.mlp.experts.register_forward_hook(self.output("routed_experts"))
        layer.mlp.shared_expert.register_forward_hook(self.output("shared_expert"))
        layer.mlp.shared_expert_gate.register_forward_hook(self.output("shared_gate"))
        layer.register_forward_hook(self.finish)
        for group, head in enumerate(predictor_heads, start=1):
            head.register_forward_hook(self.predictor_output(group))

    def begin(
        self,
        layer: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        named_inputs: dict[str, torch.Tensor | ForwardBatch | None],
    ) -> None:
        forward_batch = named_inputs["forward_batch"]
        assert isinstance(forward_batch, ForwardBatch)
        if not forward_batch.forward_mode.is_decode():
            self.decode_snapshot = None
            return
        else:
            pass
        assert self.active is None
        self.active = {}
        self.decode_snapshot = self.active
        hidden_states = named_inputs["hidden_states"]
        positions = named_inputs["positions"]
        assert isinstance(hidden_states, torch.Tensor)
        assert isinstance(positions, torch.Tensor)
        self.record("contribution", hidden_states)
        self.record("positions", positions.movedim(-1, 0))
        previous_residual = residual_batch.written_residual(forward_batch)
        if previous_residual is not None:
            self.record("previous_residual", previous_residual)
        else:
            pass
        if torch.cuda.is_current_stream_capturing():
            self.snapshots.append(self.active)
        else:
            self.eager = self.active

    def record(self, name: str, tensor: torch.Tensor) -> None:
        if self.active is not None:
            self.active[name] = tensor.detach().clone()
        else:
            pass

    def attention_input(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        named_inputs: dict[str, torch.Tensor | ForwardBatch],
    ) -> None:
        if self.active is None:
            return
        else:
            pass
        hidden_states = named_inputs["hidden_states"]
        forward_batch = named_inputs["forward_batch"]
        assert isinstance(hidden_states, torch.Tensor)
        assert isinstance(forward_batch, ForwardBatch)
        self.record("attention_normalized", hidden_states)
        residual = residual_batch.written_residual(forward_batch)
        assert residual is not None
        self.record("attention_residual", residual)

    def qkv_after_rope(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardBatch],
    ) -> None:
        query, key, value, forward_batch = inputs
        self.record("query_after_rope", query)
        self.record("key_after_rope", key)
        self.record("value_before_cache", value)

    def before_rope(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        positions, query, key = inputs
        self.record("query_before_rope", query)
        self.record("key_before_rope", key)

    def feed_forward_input(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ForwardBatch]
    ) -> None:
        if self.active is None:
            return
        else:
            pass
        hidden_states, forward_batch = inputs
        self.record("mlp_normalized", hidden_states)
        residual = residual_batch.written_residual(forward_batch)
        assert residual is not None
        self.record("mlp_residual", residual)

    def router_selection(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: TopKOutput
    ) -> None:
        if self.active is not None:
            assert TopKOutputChecker.format_is_standard(output)
            self.record("expert_ids", output.topk_ids)
            self.record("expert_weights", output.topk_weights)
        else:
            pass

    def output(self, name: str) -> OutputRecorder:
        def record_output(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        ) -> None:
            if self.active is not None:
                values = output[0] if isinstance(output, tuple) else output
                self.record(name, values)
            else:
                pass

        return record_output

    def finish(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self.record("layer_output", output)
        self.active = None

    def predictor_output(self, group: int) -> OutputRecorder:
        def record_output(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        ) -> None:
            if self.decode_snapshot is not None:
                assert isinstance(output, tuple)
                self.decode_snapshot[f"predictor_logits_{group}"] = (
                    output[0].detach().clone()
                )
                if group == self.predictor_groups:
                    self.decode_snapshot = None
                else:
                    pass
            else:
                pass

        return record_output

    def snapshot_for_batch(
        self, graph_rows: int, graph_replay: bool
    ) -> dict[str, torch.Tensor]:
        if graph_replay:
            matches = [
                snapshot
                for snapshot in self.snapshots
                if snapshot["contribution"].shape[0] == graph_rows
            ]
            assert len(matches) == 1, (graph_rows, len(matches))
            return matches[0]
        else:
            return self.eager

    def read(
        self,
        row: int,
        graph_rows: int,
        graph_replay: bool,
        include_peer_operators: bool = False,
    ) -> dict[str, torch.Tensor]:
        source = self.snapshot_for_batch(graph_rows, graph_replay)
        result = {
            f"layer{self.layer_index}_{name}": values[row].detach().clone()
            for name, values in source.items()
            if not name.startswith("predictor_logits_")
        }
        peer_fields = (
            tuple(name for name in source if not name.startswith("predictor_logits_"))
            if include_peer_operators
            else ("mlp_normalized", "expert_ids", "expert_weights")
        )
        for name in peer_fields:
            result[f"layer{self.layer_index}_peer_{name}"] = (
                source[name].detach().clone()
            )
        return result
