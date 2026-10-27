# SPDX-License-Identifier: Apache-2.0
"""Device graphs for the Qwen3-TTS reference encoder at bucketed lengths."""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

import torch
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiConv1d

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph
from sglang_omni.utils.device import device_guard

logger = logging.getLogger(__name__)

# note(ratish): a replay costs a floor plus a small per frame term and a key costs
# 32 MiB, so keys below 32 frames save nothing measurable; the step keeps the padding
# under half a clip, and clips past 256 frames run eager.
DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES = (32, 48, 64, 96, 128, 192, 256)


class ReferenceEncoderGraphStats(TypedDict):
    enabled: bool
    disable_reason: str | None
    bucket_frames: list[int]
    captured: list[int]
    replays: int
    misses: int


def move_conv_padding_to_host(encoder: torch.nn.Module) -> int:
    """Put every conv's padding integers on the CPU; returns the conv count."""
    count = 0
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.stride = module.stride.cpu()
            module.kernel_size = module.kernel_size.cpu()
            module.padding_total = module.padding_total.cpu()
            module.padding_right = module.padding_total // 2
            module.padding_left = module.padding_total - module.padding_right
            count += 1
        else:
            pass
    return count


def pad_tail(
    encoder: MimiModel,
    strides_before: int,
    module: MimiConv1d,
    args: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Fill a strided conv's input past the clip's span as the conv pads the clip."""
    hidden = args[0]
    span = (encoder.clip_samples + strides_before - 1) // strides_before
    inside = torch.arange(hidden.shape[-1], device=hidden.device) < span
    if module.pad_mode == "replicate":
        tail = hidden.index_select(-1, (span - 1).view(1))
    else:
        tail = 0.0
    return (torch.where(inside, hidden, tail), *args[1:])


def pad_strided_conv_tails(encoder: MimiModel) -> None:
    """Make a padded encode give the clip's exact-length codes.

    Every encode first sets the clip's samples in encoder.clip_samples.
    """
    # note (ratish): at the exact length each strided conv pads past the clip in its own
    # mode, and a stride-s conv yields ceil(L / s) frames, so the clip spans its samples
    # over the strides before the conv, rounded up.
    device = next(encoder.parameters()).device
    encoder.register_buffer(
        "clip_samples",
        torch.zeros((), dtype=torch.long, device=device),
        persistent=False,
    )
    convs = [m for m in encoder.encoder.modules() if isinstance(m, MimiConv1d)]
    strides_before = 1
    for conv in [*convs, encoder.downsample]:
        stride = int(conv.stride)
        if stride == 1:
            continue
        elif conv.pad_mode not in ("constant", "replicate"):
            raise ValueError(
                f"Qwen3-TTS reference encoder cannot pad a {conv.pad_mode!r} conv's tail"
            )
        else:
            conv.register_forward_pre_hook(
                functools.partial(pad_tail, encoder, strides_before)
            )
            strides_before *= stride


def smallest_bucket(frames: int, buckets: Iterable[int]) -> int | None:
    """The smallest bucket at or above frames; None if none is."""
    fitting = [bucket for bucket in buckets if bucket >= frames]
    return min(fitting) if fitting else None


@dataclass
class CapturedEncoderGraph:
    graph: ReplayableGraph
    static_input: torch.Tensor
    static_codes: torch.Tensor


class Qwen3TTSReferenceEncoderCudaGraphRunner:
    """One captured encode per bucket length, batch 1, replayed on the current stream."""

    def __init__(
        self,
        encoder: MimiModel,
        *,
        hop: int,
        num_quantizers: int,
        bucket_frames: Iterable[int],
        stream: torch.Stream,
    ) -> None:
        self.encoder = encoder
        self.hop = int(hop)
        self.num_quantizers = int(num_quantizers)
        self.bucket_frames = tuple(sorted({int(f) for f in bucket_frames}))
        self.stream = stream
        param = next(encoder.parameters())
        self.device = param.device
        self.device_module = torch.get_device_module(self.device)
        self.graph_backend: DeviceGraphBackend | None = (
            current_platform.get_device_graph_backend(self.device)
        )
        self.dtype = param.dtype
        self.graphs: dict[int, CapturedEncoderGraph] = {}
        self.pool: tuple[int, int] | None = None
        self.disable_reason: str | None = None
        self.replays = 0
        self.misses = 0

    def capture(self) -> None:
        if self.graph_backend is None:
            self.disable_reason = f"no graph backend on {self.device.type}"
            return
        else:
            pass
        graphs: dict[int, CapturedEncoderGraph] = {}
        try:
            with device_guard(self.device):
                pool = self.graph_backend.graph_pool_handle()
                # note(ratish): largest first so the shared pool is sized once.
                for frames in reversed(self.bucket_frames):
                    graphs[frames] = self.capture_bucket(frames, pool)
        except Exception as exc:
            self.disable_reason = f"capture_failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "Qwen3-TTS reference encoder graph capture disabled the runner: %s",
                self.disable_reason,
                exc_info=True,
            )
            return
        self.graphs = {frames: graphs[frames] for frames in self.bucket_frames}
        self.pool = pool
        logger.info(
            "Qwen3-TTS reference encoder graphs captured for %s frames",
            list(self.bucket_frames),
        )

    def capture_bucket(
        self, frames: int, pool: tuple[int, int]
    ) -> CapturedEncoderGraph:
        static_input = torch.zeros(
            (1, 1, frames * self.hop), device=self.device, dtype=self.dtype
        )
        self.encoder.clip_samples.fill_(frames * self.hop)
        self.stream.wait_stream(self.device_module.current_stream(self.device))
        with torch.inference_mode(), self.device_module.stream(self.stream):
            for _ in range(2):
                self._encode(static_input)
        with (
            torch.inference_mode(),
            self.graph_backend.capture(
                pool=pool,
                stream=self.stream,
                thread_local_errors=True,
            ) as graph,
        ):
            static_codes = self._encode(static_input)
        self.stream.synchronize()
        return CapturedEncoderGraph(
            graph=graph, static_input=static_input, static_codes=static_codes
        )

    def _encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(
            values, num_quantizers=self.num_quantizers, return_dict=True
        ).audio_codes

    def bucket_for(self, frames: int) -> int | None:
        return smallest_bucket(frames, self.graphs)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor | None:
        """Codes (frames, quantizers) of a waveform (samples,); None above the largest bucket."""
        samples = waveform.numel()
        frames = -(-samples // self.hop)
        bucket = self.bucket_for(frames)
        if bucket is None:
            self.misses += 1
            return None
        else:
            pass
        captured = self.graphs[bucket]
        # note (ratish): the tail padding hooks keep the last clip's samples past this one out of every code.
        self.encoder.clip_samples.fill_(samples)
        captured.static_input[0, 0, :samples].copy_(waveform)
        captured.graph.replay()
        self.replays += 1
        # note(ratish): the graph rewrites its output on the next replay.
        return captured.static_codes[0, :, :frames].transpose(0, 1).clone()

    def stats(self) -> ReferenceEncoderGraphStats:
        return {
            "enabled": bool(self.graphs),
            "disable_reason": self.disable_reason,
            "bucket_frames": list(self.bucket_frames),
            "captured": list(self.graphs),
            "replays": self.replays,
            "misses": self.misses,
        }


__all__ = [
    "DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES",
    "Qwen3TTSReferenceEncoderCudaGraphRunner",
    "move_conv_padding_to_host",
    "pad_strided_conv_tails",
    "smallest_bucket",
]
