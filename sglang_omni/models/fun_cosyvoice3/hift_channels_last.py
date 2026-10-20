# SPDX-License-Identifier: Apache-2.0
"""HiFT's decode over channels-last activations: cuDNN's NHWC convolution engines without
their layout transposes, with the Snake and spectrum kernels, as the object the vocoder
installs in place of the module's decode."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.hift_kernels import (
    overlap_add_inverse,
    short_time_spectrum,
    snake,
)
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_weight,
)


class ChannelsLastHiftDecode:
    """CosyVoice's decode written over (rows, samples, channels) activations: the
    finalized call, and the non-final one that keeps the look-right frames as context.
    """

    def __init__(self, hift: torch.nn.Module, device: torch.device) -> None:
        self.hift = hift
        self.spectrum_frames_per_mel_frame = math.prod(hift.upsample_rates)
        blocks = [*hift.source_resblocks, *hift.resblocks]
        convs = [hift.conv_pre, *hift.ups, *hift.source_downs, hift.conv_post]
        for block in blocks:
            convs.extend((*block.convs1, *block.convs2))
        # note (ratish): copies with channels-last strides; the modules keep their own
        # for the devices that run the module's decode.
        self.channels_last_weights: dict[int, torch.Tensor] = {
            id(conv): channels_last_weight(conv) for conv in convs
        }
        self.snake_alphas: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for block in blocks:
            for activation in (*block.activations1, *block.activations2):
                assert not activation.alpha_logscale, "HiFT builds linear scale alphas"
                alpha = activation.alpha.detach().contiguous()
                self.snake_alphas[id(activation)] = (
                    alpha,
                    1.0 / (alpha + activation.no_div_by_zero),
                )
        self.n_fft = hift.istft_params["n_fft"]
        self.hop_length = hift.istft_params["hop_len"]
        self.bins = self.n_fft // 2 + 1
        window = hift.stft_window.to(device=device, dtype=torch.float64)
        angle = (2 * math.pi / self.n_fft) * torch.outer(
            torch.arange(self.bins, device=device, dtype=torch.float64),
            torch.arange(self.n_fft, device=device, dtype=torch.float64),
        )
        self.forward_basis = (
            torch.cat([torch.cos(angle), -torch.sin(angle)]) * window
        ).float()
        # note (ratish): a one sided inverse counts the bins between DC and Nyquist twice.
        one_sided = torch.full((self.bins, 1), 2.0, device=device, dtype=torch.float64)
        one_sided[0] = 1.0
        one_sided[-1] = 1.0
        self.inverse_basis = (
            torch.cat([torch.cos(angle) * one_sided, -torch.sin(angle) * one_sided])
            * window
            / self.n_fft
        ).float()
        self.squared_window = hift.stft_window.to(device).pow(2)

    def spectrum(self, signal: torch.Tensor) -> torch.Tensor:
        """The centered short time spectrum of (rows, samples): (rows, 2 * bins, frames),
        the real parts then the imaginary parts."""
        half = self.n_fft // 2
        padded_signal = F.pad(signal[:, None], (half, half), mode="reflect")[
            :, 0
        ].contiguous()
        return short_time_spectrum(padded_signal, self.forward_basis, self.hop_length)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """The inverse transform of (rows, bins, frames) magnitudes and phases."""
        magnitude = torch.clip(magnitude, max=1e2)
        return overlap_add_inverse(
            (magnitude * torch.cos(phase)).contiguous(),
            (magnitude * torch.sin(phase)).contiguous(),
            self.inverse_basis,
            self.squared_window,
            self.hop_length,
        )

    def conv(
        self,
        module: torch.nn.Conv1d,
        hidden_states: torch.Tensor,
        *,
        left_padding: int,
        right_padding: int = 0,
        stride: int = 1,
    ) -> torch.Tensor:
        """The conv of (rows, samples, channels) with zeros for its causal context."""
        length = hidden_states.shape[1]
        alignment = (-(left_padding + length + right_padding)) % module.dilation[0]
        padded = F.pad(hidden_states, (0, 0, left_padding, right_padding + alignment))
        return channels_last_conv1d(
            padded,
            module,
            self.channels_last_weights[id(module)],
            length,
            stride=stride,
        )

    def snake(
        self, activation: torch.nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        alpha, inverse_alpha = self.snake_alphas[id(activation)]
        return snake(hidden_states, alpha, inverse_alpha)

    def resblock(
        self, block: torch.nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        for activation1, conv1, activation2, conv2 in zip(
            block.activations1,
            block.convs1,
            block.activations2,
            block.convs2,
            strict=True,
        ):
            residual = self.snake(activation1, hidden_states)
            residual = self.conv(conv1, residual, left_padding=conv1.causal_padding)
            residual = self.snake(activation2, residual)
            residual = self.conv(conv2, residual, left_padding=conv2.causal_padding)
            hidden_states = residual + hidden_states
        return hidden_states

    def __call__(
        self, x: torch.Tensor, s: torch.Tensor, finalize: bool = True
    ) -> torch.Tensor:
        hift = self.hift
        # note (ratish): three small transposes, the mel, the source spectrum and the
        # output spectrum, in place of one per convolution inside cuDNN.
        source_spectrum = self.spectrum(s.squeeze(1)).transpose(1, 2).contiguous()
        if finalize:
            right_padding = hift.conv_pre.causal_padding
        else:
            right_padding = 0
            held_frames = self.spectrum_frames_per_mel_frame * hift.conv_pre_look_right
            source_spectrum = source_spectrum[:, :-held_frames].contiguous()
        hidden_states = self.conv(
            hift.conv_pre,
            x.transpose(1, 2).contiguous(),
            left_padding=0,
            right_padding=right_padding,
        )
        last_stage = hift.num_upsamples - 1
        for stage in range(hift.num_upsamples):
            upsample = hift.ups[stage]
            hidden_states = F.leaky_relu(hidden_states, hift.lrelu_slope)
            hidden_states = self.conv(
                upsample,
                hidden_states.repeat_interleave(
                    int(upsample.upsample.scale_factor), dim=1
                ),
                left_padding=upsample.causal_padding,
            )
            if stage == last_stage:
                # note (ratish): the reflection pad of one sample is the second sample.
                hidden_states = torch.cat((hidden_states[:, 1:2], hidden_states), dim=1)
            else:
                pass
            source_down = hift.source_downs[stage]
            source = self.conv(
                source_down,
                source_spectrum,
                left_padding=source_down.causal_padding,
                stride=source_down.stride[0],
            )
            hidden_states = hidden_states + self.resblock(
                hift.source_resblocks[stage], source
            )
            blocks = hift.resblocks[
                stage * hift.num_kernels : (stage + 1) * hift.num_kernels
            ]
            summed = self.resblock(blocks[0], hidden_states)
            for block in blocks[1:]:
                summed = summed + self.resblock(block, hidden_states)
            hidden_states = summed / hift.num_kernels
        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv(
            hift.conv_post, hidden_states, left_padding=hift.conv_post.causal_padding
        )
        magnitude = torch.exp(hidden_states[..., : self.bins]).transpose(1, 2)
        phase = torch.sin(hidden_states[..., self.bins :]).transpose(1, 2)
        audio = self.inverse(magnitude, phase)
        if finalize:
            pass
        else:
            audio = audio[:, : -(self.spectrum_frames_per_mel_frame * self.hop_length)]
        return torch.clamp(audio, -hift.audio_limit, hift.audio_limit)
