# SPDX-License-Identifier: Apache-2.0
"""Fixed-grid AdaLN modulation values for the CosyVoice3 Flow DiT."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

BLOCK_MODULATION_PARTS = 6
FINAL_MODULATION_PARTS = 2

TimestepModulationParameterSignature = tuple[
    tuple[int, int, torch.dtype, torch.device, tuple[int, ...]], ...
]


def apply_block_modulation(
    normalization: torch.nn.Module,
    hidden_states: torch.Tensor,
    modulation: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    shift, scale, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(
        BLOCK_MODULATION_PARTS, dim=1
    )
    normalized = normalization.norm(hidden_states) * (1 + scale[:, None])
    normalized = normalized + shift[:, None]
    return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


def apply_final_modulation(
    normalization: torch.nn.Module,
    hidden_states: torch.Tensor,
    modulation: torch.Tensor,
) -> torch.Tensor:
    scale, shift = modulation.chunk(FINAL_MODULATION_PARTS, dim=1)
    return (
        normalization.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
    )


def euler_step_times(
    time_span: torch.Tensor, *, flow_dtype: torch.dtype, euler_steps: int
) -> tuple[torch.Tensor, ...]:
    if time_span.ndim != 1 or time_span.numel() != euler_steps + 1:
        raise ValueError(
            "time_span must contain exactly one more value than euler_steps"
        )
    elif not bool(torch.isfinite(time_span).all()) or not bool(
        (time_span[1:] > time_span[:-1]).all()
    ):
        raise ValueError("time_span must be finite and strictly increasing")
    else:
        pass

    flow_time = torch.zeros(1, device=time_span.device, dtype=flow_dtype)
    t, dt = time_span[0], time_span[1] - time_span[0]
    step_times: list[torch.Tensor] = []
    for euler_step in range(euler_steps):
        flow_time[:] = t
        step_times.append(flow_time.clone())
        t = t + dt
        if euler_step < euler_steps - 1:
            dt = time_span[euler_step + 2] - t
        else:
            pass
    return tuple(step_times)


def model_parameter_signature(
    dit: torch.nn.Module,
) -> TimestepModulationParameterSignature:
    projection_modules: list[torch.nn.Module] = [dit.time_embed]
    projection_modules.extend(
        block.attn_norm.linear for block in dit.transformer_blocks
    )
    projection_modules.append(dit.norm_out.linear)
    return tuple(
        (
            id(parameter),
            int(parameter._version),  # noqa: leading-underscore
            parameter.dtype,
            parameter.device,
            tuple(parameter.shape),
        )
        for module in projection_modules
        for parameter in module.parameters(recurse=True)
    )


@dataclass(frozen=True, kw_only=True)
class TimestepModulationCache:
    """One model's immutable AdaLN table for one fixed Euler schedule."""

    time_span: torch.Tensor
    block_modulations: torch.Tensor
    final_modulations: torch.Tensor
    model_id: int
    parameter_signature: TimestepModulationParameterSignature
    flow_dtype: torch.dtype

    @classmethod
    def build(
        cls,
        dit: torch.nn.Module,
        time_span: torch.Tensor,
        *,
        flow_dtype: torch.dtype,
        euler_steps: int,
    ) -> TimestepModulationCache:
        step_times = euler_step_times(
            time_span, flow_dtype=flow_dtype, euler_steps=euler_steps
        )
        block_modulations: list[torch.Tensor] = []
        final_modulations: list[torch.Tensor] = []
        autocast_enabled = time_span.device.type == "cuda" and flow_dtype in (
            torch.float16,
            torch.bfloat16,
        )
        autocast_context = (
            torch.autocast(
                device_type=time_span.device.type,
                dtype=flow_dtype,
            )
            if autocast_enabled
            else nullcontext()
        )
        with (
            torch.inference_mode(),
            autocast_context,
        ):
            for step_time in step_times:
                timestep_embedding = dit.time_embed(step_time)
                block_modulations.append(
                    torch.stack(
                        [
                            block.attn_norm.linear(
                                block.attn_norm.silu(timestep_embedding)
                            )
                            for block in dit.transformer_blocks
                        ]
                    )
                )
                final_modulations.append(
                    dit.norm_out.linear(dit.norm_out.silu(timestep_embedding))
                )
        return cls(
            time_span=time_span.detach().clone(),
            block_modulations=torch.stack(block_modulations).detach().contiguous(),
            final_modulations=torch.stack(final_modulations).detach().contiguous(),
            model_id=id(dit),
            parameter_signature=model_parameter_signature(dit),
            flow_dtype=flow_dtype,
        )

    def validate(self, dit: torch.nn.Module, time_span: torch.Tensor) -> None:
        if (
            id(dit) != self.model_id
            or model_parameter_signature(dit) != self.parameter_signature
        ):
            raise RuntimeError("CosyVoice3 timestep modulation cache is stale")
        elif (
            time_span.device != self.time_span.device
            or time_span.dtype != self.time_span.dtype
            or time_span.shape != self.time_span.shape
        ):
            raise RuntimeError(
                "CosyVoice3 timestep modulation cache has an incompatible schedule"
            )
        elif time_span is self.time_span:
            return
        elif not torch.equal(time_span, self.time_span):
            raise RuntimeError(
                "CosyVoice3 timestep modulation cache requires the canonical schedule"
            )
        else:
            pass
