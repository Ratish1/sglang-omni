# SPDX-License-Identifier: Apache-2.0
"""Device graphs for the Qwen3-TTS speaker encoder at bucketed mel lengths."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
    ReferenceEncoderGraphStats,
    smallest_bucket,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.utils.device import device_guard, supports_device_streams

logger = logging.getLogger(__name__)

SPEAKER_MEL_N_FFT = 1024
SPEAKER_MEL_HOP = 256
SPEAKER_MEL_FMIN = 0
SPEAKER_MEL_FMAX = 12000
SPEAKER_MEL_CLIP = 1e-5


def reflect_index(length: torch.Tensor, width: int, pad: int) -> torch.Tensor:
    """Gather index of a reflect pad of pad frames around the first length frames
    of a width frame buffer."""
    position = torch.arange(-pad, width + pad, device=length.device)
    index = position.abs()
    over = index - (length - 1)
    index = torch.where(over > 0, (length - 1) - over, index)
    return index.clamp(0, width - 1)


def log_mel(
    samples: torch.Tensor,
    length: torch.Tensor,
    window: torch.Tensor,
    mel_basis: torch.Tensor,
) -> torch.Tensor:
    """Log mel (1, mels, frames) of the first length samples of a (1, width) buffer, the
    checkpoint's front end: its reflect pad built around the true end, so the frames that
    lie within the length read only those samples."""
    padding = (SPEAKER_MEL_N_FFT - SPEAKER_MEL_HOP) // 2
    padded = samples.index_select(1, reflect_index(length, samples.shape[1], padding))
    spectrum = torch.stft(
        padded,
        SPEAKER_MEL_N_FFT,
        hop_length=SPEAKER_MEL_HOP,
        win_length=SPEAKER_MEL_N_FFT,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
    return torch.log(
        torch.clamp(torch.matmul(mel_basis, magnitude), min=SPEAKER_MEL_CLIP)
    )


def mel_frames(samples: int) -> int:
    """The log mel's frame count for a clip of samples."""
    padding = (SPEAKER_MEL_N_FFT - SPEAKER_MEL_HOP) // 2
    return (samples + 2 * padding - SPEAKER_MEL_N_FFT) // SPEAKER_MEL_HOP + 1


def time_delay_block(
    block: torch.nn.Module, x: torch.Tensor, indices: dict[int, torch.Tensor]
) -> torch.Tensor:
    conv = block.conv
    pad = conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
    if pad:
        x = x.index_select(2, indices[pad])
    else:
        pass
    return F.relu(F.conv1d(x, conv.weight, conv.bias, dilation=conv.dilation))


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    """Mean of x (1, channels, width) over its first length frames, accumulated and
    divided in the precision torch.mean uses and rounded to the dtype of x once."""
    accumulate = torch.promote_types(x.dtype, torch.float32)
    return ((x * mask).sum(2, keepdim=True, dtype=accumulate) / length).to(x.dtype)


def encode_bucketed(
    encoder: torch.nn.Module,
    mels: torch.Tensor,
    length: torch.Tensor,
    pads: Iterable[int],
) -> torch.Tensor:
    """The encoder's forward over its own weights, exact for the first length frames
    of mels (1, mels, width) whatever fills the rest of the width."""
    width = mels.shape[2]
    indices = {pad: reflect_index(length, width, pad) for pad in pads}
    mask = (torch.arange(width, device=mels.device) < length).to(mels.dtype)[None, None]
    x = time_delay_block(encoder.blocks[0], mels, indices)
    outputs = []
    for layer in encoder.blocks[1:]:
        residual = x
        x = time_delay_block(layer.tdnn1, x, indices)
        first, *rest = torch.chunk(x, layer.res2net_block.scale, dim=1)
        parts = [first]
        for block, part in zip(layer.res2net_block.blocks, rest):
            if len(parts) > 1:
                part = part + parts[-1]
            else:
                pass
            parts.append(time_delay_block(block, part, indices))
        x = time_delay_block(layer.tdnn2, torch.cat(parts, dim=1), indices)
        se = layer.se_block
        mean = masked_mean(x, mask, length)
        x = x * torch.sigmoid(se.conv2(F.relu(se.conv1(mean)))) + residual
        outputs.append(x)
    x = time_delay_block(encoder.mfa, torch.cat(outputs, dim=1), indices)
    asp = encoder.asp
    statistics = (
        asp._compute_statistics
    )  # noqa: leading-underscore  # upstream spelling
    mean, std = statistics(x, mask / length.to(x.dtype))
    attention = torch.cat(
        [
            x,
            mean.unsqueeze(2).expand(-1, -1, width),
            std.unsqueeze(2).expand(-1, -1, width),
        ],
        dim=1,
    )
    attention = asp.conv(torch.tanh(time_delay_block(asp.tdnn, attention, indices)))
    attention = F.softmax(attention.masked_fill(mask == 0, float("-inf")), dim=2)
    mean, std = statistics(x, attention)
    return encoder.fc(torch.cat((mean, std), dim=1).unsqueeze(2)).squeeze(-1)[0]


@dataclass
class CapturedSpeakerEncoderGraph:
    graph: ReplayableGraph
    static_samples: torch.Tensor
    static_sample_count: torch.Tensor
    static_frames: torch.Tensor
    static_embedding: torch.Tensor
    # note (ratish): the graph reads these two by address, so it holds them.
    mel_window: torch.Tensor
    mel_basis: torch.Tensor


class Qwen3TTSSpeakerEncoderCudaGraphRunner:
    """The speaker embedding of a waveform: one captured mel front end and encode per
    bucket length, the checkpoint's mel on the CPU and an eager encode past the largest
    bucket."""

    def __init__(self, encoder: torch.nn.Module, *, sample_rate: int) -> None:
        self.encoder = encoder
        self.mel_basis = torch.from_numpy(
            librosa_mel_fn(
                sr=sample_rate,
                n_fft=SPEAKER_MEL_N_FFT,
                n_mels=encoder.blocks[0].conv.in_channels,
                fmin=SPEAKER_MEL_FMIN,
                fmax=SPEAKER_MEL_FMAX,
            )
        ).float()
        self.mel_window = torch.hann_window(
            SPEAKER_MEL_N_FFT, dtype=torch.float32, device="cpu"
        )
        self.pads = frozenset(
            conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
            for conv in encoder.modules()
            if isinstance(conv, torch.nn.Conv1d)
        ) - {0}
        self.lock = threading.Lock()
        self.graphs: dict[int, CapturedSpeakerEncoderGraph] = {}
        self.bucket_frames: list[int] = []
        self.disable_reason: str | None = None
        self.replays = 0
        self.misses = 0

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Log mel (1, mels, frames) of a CPU waveform (1, samples)."""
        return log_mel(
            waveform,
            torch.tensor([waveform.shape[1]]),
            self.mel_window,
            self.mel_basis,
        )

    def capture(self, codec_frame_buckets: Iterable[int], codec_hop: int) -> None:
        """One graph per bucket of the reference encoder's ladder, in mel frames;
        none when a capture fails, so every clip runs eager."""
        param = next(self.encoder.parameters())
        buckets = sorted(
            frames * codec_hop // SPEAKER_MEL_HOP for frames in codec_frame_buckets
        )
        self.bucket_frames = buckets
        graph_backend = current_platform.get_device_graph_backend(param.device)
        if not buckets:
            return
        elif not supports_device_streams(param.device) or graph_backend is None:
            self.disable_reason = f"no graph backend on {param.device.type}"
            return
        else:
            pass
        device_module = torch.get_device_module(param.device)
        window = self.mel_window.to(param.device)
        mel_basis = self.mel_basis.to(param.device)

        def bucket_embedding(
            samples: torch.Tensor, sample_count: torch.Tensor, frames: torch.Tensor
        ) -> torch.Tensor:
            mels = log_mel(samples, sample_count, window, mel_basis)
            return encode_bucketed(
                self.encoder, mels.to(param.dtype), frames, self.pads
            )

        graphs: dict[int, CapturedSpeakerEncoderGraph] = {}
        try:
            with device_guard(param.device):
                pool = graph_backend.graph_pool_handle()
                stream = device_module.Stream(device=param.device)
                # note(ratish): largest first so the shared pool is sized once.
                for frames in reversed(buckets):
                    # note (ratish): the widest clip whose mel has frames frames; the
                    # warmups make the FFT plans, whose creation waits for the device.
                    sample_count = SPEAKER_MEL_HOP * (frames + 1) - 1
                    static_samples = torch.zeros(
                        (1, sample_count), device=param.device, dtype=torch.float32
                    )
                    static_sample_count = torch.full(
                        (1,), sample_count, device=param.device, dtype=torch.long
                    )
                    static_frames = torch.full(
                        (1,), frames, device=param.device, dtype=torch.long
                    )
                    stream.wait_stream(device_module.current_stream(param.device))
                    with torch.inference_mode(), device_module.stream(stream):
                        for _ in range(2):
                            bucket_embedding(
                                static_samples, static_sample_count, static_frames
                            )
                    with (
                        torch.inference_mode(),
                        graph_backend.capture(
                            pool=pool,
                            stream=stream,
                            thread_local_errors=True,
                        ) as graph,
                    ):
                        static_embedding = bucket_embedding(
                            static_samples, static_sample_count, static_frames
                        )
                    stream.synchronize()
                    graphs[frames] = CapturedSpeakerEncoderGraph(
                        graph=graph,
                        static_samples=static_samples,
                        static_sample_count=static_sample_count,
                        static_frames=static_frames,
                        static_embedding=static_embedding,
                        mel_window=window,
                        mel_basis=mel_basis,
                    )
        except Exception as exc:
            self.disable_reason = f"capture_failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "Qwen3-TTS speaker encoder graph capture disabled the runner: "
                f"{self.disable_reason}",
                exc_info=True,
            )
            return
        self.graphs = graphs
        logger.info(f"Qwen3-TTS speaker encoder graphs captured for {buckets} frames")

    def stats(self) -> ReferenceEncoderGraphStats:
        return {
            "enabled": bool(self.graphs),
            "disable_reason": self.disable_reason,
            "bucket_frames": list(self.bucket_frames),
            "captured": sorted(self.graphs),
            "replays": self.replays,
            "misses": self.misses,
        }

    def embed(self, waveform: np.ndarray) -> torch.Tensor:
        """Speaker embedding (enc_dim,) of a waveform at the encoder's sample rate."""
        samples = torch.from_numpy(waveform).unsqueeze(0)
        frames = mel_frames(samples.shape[1])
        assert frames > max(self.pads)
        bucket = smallest_bucket(frames, self.graphs)
        if bucket is None:
            param = next(self.encoder.parameters())
            mels = self.mel(samples).to(param.device).to(param.dtype)
            # note (Haoling Pu): preprocessing threads embed concurrently, and
            # model_info reports this count.
            with self.lock:
                self.misses += 1
            return self.encoder(mels.transpose(1, 2))[0]
        else:
            pass
        captured = self.graphs[bucket]
        # note(ratish): every worker enqueues on one shared stream, so the lock only
        # orders the enqueue and the clone lands before the next replay.
        with self.lock:
            captured.static_samples[:, : samples.shape[1]].copy_(samples)
            captured.static_sample_count.fill_(samples.shape[1])
            captured.static_frames.fill_(frames)
            captured.graph.replay()
            self.replays += 1
            return captured.static_embedding.clone()
