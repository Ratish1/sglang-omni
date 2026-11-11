"""Retain native projection operands before mutation, including graph padding."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod,
    get_bf16_gemm_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer
from torch import nn

LAYOUT_FIELDS = (
    "rows",
    "row_stride_elements",
    "column_stride_elements",
    "storage_offset_elements",
    "data_pointer_bytes",
)


class InvariantThinkerOutputProjection(UnquantizedLinearMethod):
    """Isolate output-projection reduction geometry without global operator overrides."""

    def apply(
        self,
        layer: RowParallelLinear,
        inputs: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert inputs.is_cuda and inputs.ndim == 2
        assert inputs.dtype == layer.weight.dtype == torch.bfloat16
        assert bias is None
        return matmul_persistent(inputs, layer.weight.T)


class ThinkerBoundaryCapture:
    def __init__(
        self,
        query_indices: torch.Tensor | None,
        input_rows: torch.Tensor,
        output_rows: torch.Tensor,
        observed: torch.Tensor,
        row_layout: torch.Tensor | None = None,
    ) -> None:
        self.query_indices = query_indices
        self.input_rows = input_rows
        self.output_rows = output_rows
        self.observed = observed
        self.row_layout = row_layout

    def capture_input(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor] | tuple[torch.Tensor, ForwardBatch | None],
    ) -> None:
        tensor = arguments[0]
        if tensor.shape[0] == 0:
            return
        else:
            pass
        assert tensor.dtype == self.input_rows.dtype
        if self.query_indices is None:
            assert self.row_layout is not None
            assert tensor.shape[0] <= self.input_rows.shape[0]
            self.input_rows[: tensor.shape[0]].copy_(tensor)
            for index, value in enumerate(
                (
                    tensor.shape[0],
                    *tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                )
            ):
                self.row_layout[index].fill_(value)
        else:
            self.input_rows.copy_(tensor[self.query_indices])
        self.observed.fill_(1)

    def __call__(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor] | tuple[torch.Tensor, ForwardBatch | None],
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        if tensor.shape[0] == 0:
            return
        else:
            pass
        assert tensor.dtype == self.output_rows.dtype
        if self.query_indices is None:
            assert tensor.shape[0] <= self.output_rows.shape[0]
            self.output_rows[: tensor.shape[0]].copy_(tensor)
        else:
            self.output_rows.copy_(tensor[self.query_indices])
        self.observed.add_(1)


class FullThinkerProjectionCapture:
    def __init__(
        self,
        directory: Path,
        layers: nn.ModuleList,
        prototype: torch.Tensor,
        selected_layers: str,
    ) -> None:
        layer_indices = [int(value) for value in selected_layers.split(",")]
        if len(set(layer_indices)) != len(layer_indices) or not all(
            0 <= index < len(layers) for index in layer_indices
        ):
            raise ValueError(
                "Full projection layers must be distinct valid layer indices"
            )
        else:
            pass
        self.directory = directory
        self.modules: dict[str, RowParallelLinear] = {}
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.layouts: dict[str, torch.Tensor] = {}
        self.row_layouts: dict[str, dict[str, int]] = {}
        self.observed = torch.zeros(
            len(layer_indices), device=prototype.device, dtype=torch.int32
        )
        for slot, layer_index in enumerate(layer_indices):
            layer = layers[layer_index]
            assert isinstance(layer, Qwen3MoeDecoderLayer)
            projection = layer.self_attn.o_proj
            assert isinstance(projection, RowParallelLinear)
            name = f"layer.{layer_index}.attention_projection"
            self.modules[name] = projection
            self.inputs[name] = prototype.new_zeros(
                (prototype.shape[0], layer.self_attn.q_size)
            )
            self.outputs[name] = prototype.new_zeros(
                (prototype.shape[0], layer.self_attn.hidden_size)
            )
            self.layouts[name] = torch.zeros(
                len(LAYOUT_FIELDS), device=prototype.device, dtype=torch.int64
            )
            capture = ThinkerBoundaryCapture(
                None,
                self.inputs[name],
                self.outputs[name],
                self.observed[slot],
                self.layouts[name],
            )
            projection.register_forward_pre_hook(capture.capture_input)
            projection.register_forward_hook(capture)

    def snapshot(self, step: int, live_rows: int) -> dict[str, torch.Tensor]:
        assert bool((self.observed == 2).all().cpu())
        snapshots: dict[str, torch.Tensor] = {}
        for name, layout in self.layouts.items():
            values = layout.cpu().tolist()
            self.row_layouts[name] = dict(zip(LAYOUT_FIELDS, values, strict=True))
            row_count = self.row_layouts[name]["rows"]
            assert live_rows <= row_count <= self.inputs[name].shape[0]
            snapshots[f"{name}.input"] = self.inputs[name][:row_count].cpu().clone()
            snapshots[f"{name}.output"] = self.outputs[name][:row_count].cpu().clone()
        assert all(bool(tensor.isfinite().all()) for tensor in snapshots.values())
        if step == 0:
            weights = {
                name: module.weight.detach().cpu().clone()
                for name, module in self.modules.items()
            }
            torch.save(weights, self.directory / "actual-projection-weights.pt")
            runtime = {
                "bf16_gemm_backend": get_bf16_gemm_backend().value,
                "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "preferred_blas_library": str(
                    torch.backends.cuda.preferred_blas_library()
                ),
                "quantization_methods": {
                    name: type(module.quant_method).__name__
                    for name, module in self.modules.items()
                },
                "weight_shapes": {
                    name: list(module.weight.shape)
                    for name, module in self.modules.items()
                },
                "weight_strides": {
                    name: list(module.weight.stride())
                    for name, module in self.modules.items()
                },
                "weight_storage_offsets": {
                    name: module.weight.storage_offset()
                    for name, module in self.modules.items()
                },
                "weight_data_pointers": {
                    name: module.weight.data_ptr()
                    for name, module in self.modules.items()
                },
            }
            (self.directory / "projection-runtime.json").write_text(
                json.dumps(runtime, indent=2)
            )
        else:
            pass
        return snapshots
