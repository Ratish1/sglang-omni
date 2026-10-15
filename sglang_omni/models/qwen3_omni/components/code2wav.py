# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni code2wav whose convolutions run channels last and whose pre-transformer runs on
fused kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.layernorm import rmsnorm
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavDecoderResidualUnit,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
    SnakeBeta,
)

from sglang_omni.platforms.interface import JointRopeInplaceKernel
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
    channels_last_weight,
)
from sglang_omni.utils.incremental_codec import (
    IncrementalCodecState,
    IncrementalCodecStateSpec,
    conv_stack_state_layout,
    incremental_conv_stack,
    incremental_transformer_layers,
    init_incremental_state,
)
from sglang_omni.utils.snake_beta import FusedSnakeBeta


def causal_conv(
    module: Qwen3OmniMoeCausalConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal conv of a (B, L, C) activation, returned as (B, L, C_out)."""
    conv = module.conv
    assert conv.stride == (1,), "code2wav causal convs are stride 1"
    length = hidden_states.shape[1]
    padded = F.pad(hidden_states, (0, 0, module.padding, (-length) % conv.dilation[0]))
    return channels_last_conv1d(padded, conv, conv.weight, length)


def causal_transconv(
    module: Qwen3OmniMoeCausalTransConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal transposed conv of a (B, L, C) activation, returned as (B, L_out, C_out)."""
    conv = module.conv
    output = channels_last_conv_transpose1d(hidden_states, conv, conv.weight, conv.bias)
    return output[:, module.left_pad : output.shape[1] - module.right_pad].contiguous()


def channels_last_block(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Run one code2wav module on a (B, L, C) activation as its forward runs on (B, C, L)."""
    if isinstance(module, Qwen3OmniMoeCausalConvNet):
        return causal_conv(module, hidden_states)
    elif isinstance(module, Qwen3OmniMoeCausalTransConvNet):
        return causal_transconv(module, hidden_states)
    elif isinstance(module, (SnakeBeta, FusedSnakeBeta)):
        return module(hidden_states.transpose(1, 2)).transpose(1, 2)
    elif isinstance(module, Qwen3OmniMoeConvNeXtBlock):
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel,
        # faster than cuDNN's channels-last depthwise and free of its per-shape setup.
        normed = module.norm(
            module.dwconv(hidden_states.transpose(1, 2)).transpose(1, 2)
        )
        return hidden_states + module.gamma * module.pwconv2(
            module.act(module.pwconv1(normed))
        )
    elif isinstance(module, Qwen3OmniMoeCode2WavDecoderResidualUnit):
        output = hidden_states
        for block in (module.act1, module.conv1, module.act2, module.conv2):
            output = channels_last_block(block, output)
        return output + hidden_states
    elif isinstance(module, Qwen3OmniMoeCode2WavDecoderBlock):
        for block in module.block:
            hidden_states = channels_last_block(block, hidden_states)
        return hidden_states
    else:
        raise TypeError(
            f"no channels-last form for code2wav module {type(module).__name__}"
        )


class FusedCode2WavTransformer(torch.nn.Module):
    """Code2wav's pre-transformer on its own weights with one qkv and one gate-up GEMM per layer,
    and RMSNorm, rotary, SwiGLU and layer scale plus residual one kernel each."""

    def __init__(
        self,
        transformer: Qwen3OmniMoeCode2WavTransformerModel,
        rope: JointRopeInplaceKernel,
    ) -> None:
        super().__init__()
        rotary = transformer.rotary_emb
        assert not transformer.config.attention_bias, "code2wav attention has no bias"
        assert transformer.config.hidden_act == "silu", "code2wav's MLP is SwiGLU"
        assert rotary.rope_type == "default", "the cos/sin table is built once"
        self.transformer = transformer
        self.rope = rope
        positions = torch.arange(
            transformer.config.max_position_embeddings,
            device=rotary.inv_freq.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, rotary.inv_freq.float())
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((angles.cos(), angles.sin()), dim=-1) * rotary.attention_scaling,
            persistent=False,
        )
        # note (ratish): q, k, v and gate, up become views of one matrix each, so the
        # fused GEMMs read the loaded weights without a second copy.
        for layer in transformer.layers:
            attention, mlp = layer.self_attn, layer.mlp
            projections = (attention.q_proj, attention.k_proj, attention.v_proj)
            attention.qkv_weight = torch.cat([p.weight.data for p in projections])
            start = 0
            for projection in projections:
                rows = projection.weight.shape[0]
                projection.weight.data = attention.qkv_weight[start : start + rows]
                start += rows
            mlp.gate_up_weight = torch.cat(
                (mlp.gate_proj.weight.data, mlp.up_proj.weight.data)
            )
            mlp.gate_proj.weight.data = mlp.gate_up_weight[: mlp.intermediate_size]
            mlp.up_proj.weight.data = mlp.gate_up_weight[mlp.intermediate_size :]

    def forward(self, inputs_embeds: torch.Tensor) -> BaseModelOutputWithPast:
        batch_size, length, hidden_size = inputs_embeds.shape
        head_dim = self.transformer.layers[0].self_attn.head_dim
        sliding_window = self.transformer.config.sliding_window
        positions = torch.arange(length, device=inputs_embeds.device).repeat(batch_size)
        if length <= sliding_window:
            window_mask = None
        else:
            query = torch.arange(length, device=inputs_embeds.device)[:, None]
            key = torch.arange(length, device=inputs_embeds.device)[None, :]
            window_mask = (key <= query) & (query - key < sliding_window)
        hidden_states = inputs_embeds.reshape(batch_size * length, hidden_size)
        for layer in self.transformer.layers:
            attention = layer.self_attn
            residual = hidden_states
            normed = rmsnorm(
                hidden_states,
                layer.input_layernorm.weight,
                layer.input_layernorm.variance_epsilon,
            )
            query, key, value = F.linear(normed, attention.qkv_weight).split(
                [
                    attention.q_proj.out_features,
                    attention.k_proj.out_features,
                    attention.v_proj.out_features,
                ],
                dim=-1,
            )
            query = query.view(batch_size * length, -1, head_dim)
            key = key.view(batch_size * length, -1, head_dim)
            self.rope(query, key, self.cos_sin_cache, positions, is_neox=True)
            attended = F.scaled_dot_product_attention(
                query.view(batch_size, length, -1, head_dim).transpose(1, 2),
                key.view(batch_size, length, -1, head_dim).transpose(1, 2),
                value.view(batch_size, length, -1, head_dim).transpose(1, 2),
                attn_mask=window_mask,
                is_causal=window_mask is None,
                scale=attention.scaling,
                enable_gqa=attention.num_key_value_groups > 1,
            )
            attended = attended.transpose(1, 2).reshape(batch_size * length, -1)
            hidden_states = torch.addcmul(
                residual,
                layer.self_attn_layer_scale.scale,
                F.linear(attended, attention.o_proj.weight),
            )
            residual = hidden_states
            normed = rmsnorm(
                hidden_states,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
            )
            activated = silu_and_mul(F.linear(normed, layer.mlp.gate_up_weight))
            hidden_states = torch.addcmul(
                residual,
                layer.mlp_layer_scale.scale,
                F.linear(activated, layer.mlp.down_proj.weight),
            )
        hidden_states = rmsnorm(
            hidden_states,
            self.transformer.norm.weight,
            self.transformer.norm.variance_epsilon,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states.view(batch_size, length, hidden_size)
        )


class Qwen3OmniCode2Wav(Qwen3OmniMoeCode2Wav):
    """Code2Wav that carries (B, L, C) activations once its convs are channels last."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig) -> None:
        super().__init__(config)
        self.is_channels_last = False

    def use_channels_last(self) -> None:
        """Store every conv weight channels last and run the channels-last forward."""
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                module.weight.data = channels_last_weight(module)
            else:
                pass
        self.is_channels_last = True

    def use_fused_transformer(self, rope: JointRopeInplaceKernel) -> None:
        """Run the pre-transformer on fused kernels, with rope as its rotary kernel."""
        self.pre_transformer = FusedCode2WavTransformer(self.pre_transformer, rope)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.is_channels_last:
            return super().forward(codes)
        elif codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layer of codes, "
                f"got {codes.shape[1]}"
            )
        else:
            pass
        hidden_states = self.code_embedding(codes + self.code_offset).mean(1)
        hidden_states = self.pre_transformer(
            inputs_embeds=hidden_states
        ).last_hidden_state
        for blocks in self.upsample:
            for block in blocks:
                hidden_states = channels_last_block(block, hidden_states)
        for block in self.decoder:
            hidden_states = channels_last_block(block, hidden_states)
        return hidden_states.transpose(1, 2).clamp(min=-1, max=1)


class Code2WavIncrementalDecoder:
    """Decodes a stream's new codec frames from the stream's state: a stream's frames decoded
    in any split give the waveform Qwen3OmniCode2Wav gives on all of them at once."""

    def __init__(self, model: Qwen3OmniCode2Wav) -> None:
        pre_transformer = model.pre_transformer
        self.model = model
        if isinstance(pre_transformer, FusedCode2WavTransformer):
            self.transformer = pre_transformer.transformer
        else:
            self.transformer = pre_transformer
        self.total_upsample = int(model.total_upsample)
        # note (ratish): the conv weights are already channels last in place, so the steps
        # read the module's own weights; the depthwise ConvNeXt conv keeps its (B, C, L) path.
        if model.is_channels_last:
            self.channels_last_weights: dict[str, torch.Tensor] | None = {
                "decoder.0": model.decoder[0].conv.weight,
                "decoder.final": model.decoder[-1].conv.weight,
            }
            for stage_index, blocks in enumerate(model.upsample):
                self.channels_last_weights[f"upsample.{stage_index}.transconv"] = (
                    blocks[0].conv.weight
                )
            for block_index, block in enumerate(model.decoder[1:-2], start=1):
                self.channels_last_weights[f"decoder.{block_index}.transconv"] = (
                    block.block[1].conv.weight
                )
                for residual_index, residual in enumerate(block.block[2:]):
                    key = f"decoder.{block_index}.residual.{residual_index}"
                    self.channels_last_weights[f"{key}.conv1"] = (
                        residual.conv1.conv.weight
                    )
                    self.channels_last_weights[f"{key}.conv2"] = (
                        residual.conv2.conv.weight
                    )
        else:
            self.channels_last_weights = None
        # note (ratish): a stream's first decode drops each transposed conv's left pad once,
        # carried to the waveform through the strides after it.
        self.first_decode_shortfall = 0
        stride_after = 1
        for block in reversed(model.decoder[1:-2]):
            transconv = block.block[1]
            self.first_decode_shortfall += int(transconv.left_pad) * stride_after
            stride_after *= int(transconv.conv.stride[0])
        for blocks in reversed(model.upsample):
            self.first_decode_shortfall += int(blocks[0].left_pad) * stride_after
            stride_after *= int(blocks[0].conv.stride[0])
        conv_histories, transconv_overlaps = conv_stack_state_layout(model)
        attention = self.transformer.layers[0].self_attn
        self.spec = IncrementalCodecStateSpec(
            conv_histories=tuple(conv_histories),
            transconv_overlaps=tuple(transconv_overlaps),
            num_layers=len(self.transformer.layers),
            num_key_value_heads=int(attention.k_proj.out_features)
            // int(attention.head_dim),
            head_dim=int(attention.head_dim),
            retained_context=int(self.transformer.window_size) - 1,
        )

    def state_spec(self) -> IncrementalCodecStateSpec:
        return self.spec

    def init_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> IncrementalCodecState:
        return init_incremental_state(self.spec, batch_size, device=device, dtype=dtype)

    def decode(
        self, codes: torch.Tensor, state: IncrementalCodecState, *, first_decode: bool
    ) -> torch.Tensor:
        """The (B, 1, samples) waveform of the (B, Q, T) codes after the frames the state
        holds; every row of a first decode starts its stream."""
        fresh_frames = int(codes.shape[-1])
        hidden_states = self.model.code_embedding(codes + self.model.code_offset).mean(
            1
        )
        hidden_states = self.transformer.norm(
            incremental_transformer_layers(self.transformer, hidden_states, state, None)
        )
        if self.channels_last_weights is None:
            hidden_states = hidden_states.transpose(1, 2)
        else:
            pass
        waveform = incremental_conv_stack(
            self.model,
            hidden_states,
            state,
            self.channels_last_weights,
            first_decode=first_decode,
        )
        assert waveform.shape[-1] == fresh_frames * self.total_upsample - (
            self.first_decode_shortfall if first_decode else 0
        ), "code2wav incremental decode returned the wrong sample count"
        state.advance(fresh_frames)
        return waveform
