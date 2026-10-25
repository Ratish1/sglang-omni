"""FA3 against Triton packed attention on Qwen3-Omni's served encoder shapes: time per call and distance to fp32.

Run on the box from a tree (cwd and PYTHONPATH the tree): python3 packed_attention_kernel_bench.py
Head counts, head sizes and layer counts come from the checkpoint's config. Segment lengths are served ones: MMMU-50
image rows at the median, 75th percentile and largest h*w (920, 4536, 21440 patches, run316 grids), a Video-AMME
video's 64 frames of 22x40 patches (run316), one 390-token MMSU clip and a full 4096-token audio bucket, both cut
into 104-token windows. Each kernel gets the metadata the encoders pass. Per kernel it prints the median time of one
call over 200 eager calls back to back (launch included, as the eager layers run it) and over 200 replays of a CUDA
graph holding one call (device time alone, as the audio layer graphs run it), that time across the encoder's layers,
and the relative L2 distance to fp32 math attention computed per segment.
"""

from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.vision import (
    VisionFlash3Attention,
    VisionTritonAttention,
    prepare_vision_attention_metadata,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_WINDOW = 104
CALLS = 200
WARMUP_CALLS = 20


def windows(tokens: int) -> list[int]:
    full, remainder = divmod(tokens, AUDIO_WINDOW)
    return [AUDIO_WINDOW] * full + ([remainder] if remainder else [])


def fp32_reference(q, k, v, bounds: list[int], scale: float) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.MATH):
        return torch.cat(
            [
                F.scaled_dot_product_attention(
                    q[start:end].float().transpose(0, 1),
                    k[start:end].float().transpose(0, 1),
                    v[start:end].float().transpose(0, 1),
                    scale=scale,
                ).transpose(0, 1)
                for start, end in zip(bounds, bounds[1:])
            ]
        )


def median_ms(run, calls: int) -> float:
    times = []
    for _ in range(calls):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> None:
    thinker = AutoConfig.from_pretrained(MODEL).thinker_config
    vision, audio = thinker.vision_config, thinker.audio_config
    vision_shape = (
        vision.num_heads,
        vision.hidden_size // vision.num_heads,
        vision.depth,
    )
    audio_shape = (
        audio.encoder_attention_heads,
        audio.d_model // audio.encoder_attention_heads,
        audio.encoder_layers,
    )
    print(
        f"vision heads, head size, blocks {vision_shape}; audio heads, head size, layers {audio_shape}",
        flush=True,
    )
    cases = (
        ("vision image median", vision_shape, [920]),
        ("vision image p75", vision_shape, [4536]),
        ("vision image largest", vision_shape, [21440]),
        ("vision video 64 frames", vision_shape, [880] * 64),
        ("audio one clip", audio_shape, windows(390)),
        ("audio 4096 bucket", audio_shape, windows(4096)),
    )
    device = torch.device("cuda", 0)
    print(f"device {torch.cuda.get_device_name(device)}", flush=True)
    kernels = {"fa3": VisionFlash3Attention(), "triton": VisionTritonAttention()}
    for name, (heads, head_size, layers), segments in cases:
        torch.manual_seed(0)
        tokens = sum(segments)
        q, k, v = (
            torch.randn(tokens, heads, head_size, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        )
        cu_seqlens = (
            torch.tensor([0, *segments], dtype=torch.int32)
            .cumsum(0)
            .to(torch.int32)
            .to(device)
        )
        metadata = prepare_vision_attention_metadata(
            cu_seqlens, device, max_seqlen=max(segments)
        )
        scale = head_size**-0.5
        reference = fp32_reference(q, k, v, cu_seqlens.tolist(), scale)
        for kernel_name, kernel in kernels.items():

            def call() -> torch.Tensor:
                return kernel(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    bsz=1,
                    seq_len=tokens,
                    softmax_scale=scale,
                    forward_metadata=metadata,
                )

            with torch.no_grad():
                distance = (
                    (call().float() - reference).norm() / reference.norm()
                ).item()
                for _ in range(WARMUP_CALLS):
                    call()
                torch.cuda.synchronize()
                eager_ms = median_ms(call, CALLS)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                for _ in range(WARMUP_CALLS):
                    graph.replay()
                graph_ms = median_ms(graph.replay, CALLS)
            print(
                f"{name} | segments {len(segments)} tokens {tokens} | {kernel_name} | eager {eager_ms:.4f} ms "
                f"| graph {graph_ms:.4f} ms | graph x {layers} layers {graph_ms * layers:.3f} ms "
                f"| rel L2 to fp32 {distance:.3e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
