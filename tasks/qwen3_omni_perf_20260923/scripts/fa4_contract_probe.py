# SPDX-License-Identifier: Apache-2.0
"""FA4 against FA3 at every Qwen3-Omni site that runs FA3 on Hopper, with the checkpoint's shapes.

Runs on any GPU where SGLang's vendored FA4 has a forward (SM90 runs its SM90 kernel): the vision
and audio encoders' packed varlen attention through SGLang's VisionFlash3Attention and
VisionFlash4Attention on strided q, k and v cut from one packed projection, the audio layer graph's
zero-length padding segments included, and code2wav's later decode through the ring's page table
(page size 1, window (sliding_window - 1, 0)). Each output is compared with an fp32 reference, FA4
twice in one process for bit identity, and graph replay against eager on the same inputs.

Usage: python3 fa4_contract_probe.py CONFIG_JSON OUT_JSON
"""

from __future__ import annotations

import importlib.metadata
import json
import pathlib
import sys
import time

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.flash_attention_v4 import (
    is_flash_attention_v4_available,
)
from sglang.srt.layers.attention.vision import (
    VisionFlash3Attention,
    VisionFlash4Attention,
    VisionTritonAttention,
    prepare_vision_attention_metadata,
)
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.codec_state_pool import ring_addresses

STREAM_CHUNK_FRAMES = 10
VISION_SEGMENTS = ((1248,), (440, 224), (256, 256, 256, 256), (64,))
CODE2WAV_BATCH_SIZES = (1, 4, 16)
CODE2WAV_START_POSITIONS = (0, 10, 30, 61, 71, 200, 1009)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference = actual.float() - reference.float()
    return float(difference.norm() / reference.float().norm())


def segment_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_lengths: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    """fp32 softmax(q k^T scale) v within each segment, (tokens, heads, head_dim)."""
    outputs = []
    start = 0
    for length in segment_lengths:
        end = start + length
        q = query[start:end].float().transpose(0, 1)
        k = key[start:end].float().transpose(0, 1)
        v = value[start:end].float().transpose(0, 1)
        weights = torch.softmax(q @ k.transpose(1, 2) * scale, dim=-1)
        outputs.append((weights @ v).transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)


def packed_qkv(
    tokens: int, heads: int, head_dim: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q, k and v as the encoders cut them: strided views of one (tokens, 3, heads, head_dim) projection."""
    projection = torch.randn(
        tokens, 3, heads, head_dim, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    return projection.unbind(1)


def packed_call(
    backend: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_lengths: tuple[int, ...],
    scale: float,
    max_seqlen: int,
) -> torch.Tensor:
    cumulative = torch.tensor([0, *segment_lengths], dtype=torch.int32).cumsum(0)
    metadata = prepare_vision_attention_metadata(
        cumulative.to(torch.int32).cuda(), torch.device("cuda"), max_seqlen=max_seqlen
    )
    return backend(
        query,
        key,
        value,
        metadata.cu_seqlens,
        bsz=1,
        seq_len=query.shape[0],
        softmax_scale=scale,
        forward_metadata=metadata,
    )


def timed_first_call(call) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = call()
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


def encoder_cases(
    name: str,
    heads: int,
    head_dim: int,
    layouts: tuple[tuple[int, ...], ...],
    generator: torch.Generator,
) -> list[dict[str, object]]:
    rows = []
    backends = {
        "fa3": VisionFlash3Attention(),
        "fa4": VisionFlash4Attention(),
        "triton": VisionTritonAttention(),
    }
    scale = head_dim**-0.5
    for segment_lengths in layouts:
        tokens = sum(segment_lengths)
        query, key, value = packed_qkv(tokens, heads, head_dim, generator)
        assert query.stride(0) == 3 * heads * head_dim, "q, k and v stay strided"
        reference = segment_reference(query, key, value, segment_lengths, scale)
        max_seqlen = max(segment_lengths)
        row: dict[str, object] = {
            "site": name,
            "segments": list(segment_lengths),
            "heads": heads,
            "head_dim": head_dim,
        }
        outputs = {}
        for backend_name, backend in backends.items():
            output, seconds = timed_first_call(
                lambda: packed_call(
                    backend, query, key, value, segment_lengths, scale, max_seqlen
                )
            )
            repeat = packed_call(
                backend, query, key, value, segment_lengths, scale, max_seqlen
            )
            outputs[backend_name] = output
            row[f"{backend_name}_rel_l2_fp32"] = relative_l2(output, reference)
            row[f"{backend_name}_repeat_bit_identical"] = bool(
                torch.equal(output, repeat)
            )
            row[f"{backend_name}_first_call_s"] = round(seconds, 4)
        row["fa4_vs_fa3_rel_l2"] = relative_l2(outputs["fa4"], outputs["fa3"])
        rows.append(row)
    return rows


def audio_graph_case(
    heads: int, head_dim: int, window: int, generator: torch.Generator
) -> dict[str, object]:
    """The audio layer graph's contract on FA4: a bucket captured with window segments and
    zero-length padding segments replays a different layout of the same slot count as eager does.
    """
    bucket = 512
    slots = bucket // window + 32 + 2
    capture_layout = [window] * (bucket // window)
    if bucket % window:
        capture_layout.append(bucket % window)
    else:
        pass
    capture_layout += [0] * (slots - len(capture_layout))
    replay_real = [window, window // 2, 37]
    replay_layout = list(replay_real)
    padding = bucket - sum(replay_layout)
    replay_layout += [window] * (padding // window)
    if padding % window:
        replay_layout.append(padding % window)
    else:
        pass
    replay_layout += [0] * (slots - len(replay_layout))
    backend = VisionFlash4Attention()
    scale = head_dim**-0.5
    static_q, static_k, static_v = packed_qkv(bucket, heads, head_dim, generator)
    static_cumulative = (
        torch.tensor([0, *capture_layout], dtype=torch.int32)
        .cumsum(0)
        .to(torch.int32)
        .cuda()
    )

    def run() -> torch.Tensor:
        metadata = prepare_vision_attention_metadata(
            static_cumulative, static_cumulative.device, max_seqlen=window
        )
        return backend(
            static_q,
            static_k,
            static_v,
            metadata.cu_seqlens,
            bsz=1,
            seq_len=bucket,
            softmax_scale=scale,
            forward_metadata=metadata,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = run()
    fresh_q, fresh_k, fresh_v = packed_qkv(bucket, heads, head_dim, generator)
    static_q.copy_(fresh_q)
    static_k.copy_(fresh_k)
    static_v.copy_(fresh_v)
    static_cumulative.copy_(
        torch.tensor([0, *replay_layout], dtype=torch.int32)
        .cumsum(0)
        .to(torch.int32)
        .cuda()
    )
    graph.replay()
    replayed = static_output.clone()
    eager = run()
    nonzero = [length for length in replay_layout if length]
    reference = segment_reference(static_q, static_k, static_v, tuple(nonzero), scale)
    real_tokens = sum(replay_real)
    return {
        "site": "audio_layer_graph",
        "window": window,
        "slots": slots,
        "zero_length_segments": replay_layout.count(0),
        "replay_equals_eager": bool(torch.equal(replayed, eager)),
        "replay_rel_l2_fp32_real_tokens": relative_l2(
            replayed[:real_tokens], reference[:real_tokens]
        ),
    }


def code2wav_ring_case(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, object]:
    """code2wav's later decode: L new frames per row against the row's ring window through the
    page table, as code2wav.py calls it, on FA3 and FA4; the reference gathers the window rows
    and masks them as the non-paged path does."""
    retained = sliding_window - 1
    ring_frames = retained + STREAM_CHUNK_FRAMES
    total_slots = batch_size + 2
    length = STREAM_CHUNK_FRAMES
    width = num_kv_heads * head_dim
    keys = torch.randn(
        total_slots * ring_frames, width, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    values = torch.randn(
        total_slots * ring_frames, width, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    slot_index = torch.arange(1, batch_size + 1, device="cuda")
    start_positions = torch.tensor(
        [
            CODE2WAV_START_POSITIONS[row % len(CODE2WAV_START_POSITIONS)]
            for row in range(batch_size)
        ],
        device="cuda",
    )
    addresses = ring_addresses(
        slot_index, start_positions, length, ring_frames, retained
    )
    query = torch.randn(
        batch_size, length, num_heads, head_dim, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    page_table = addresses.window_rows.to(torch.int32)
    window_lengths = addresses.window_lengths.to(torch.int32)
    scale = head_dim**-0.5

    def paged(version: int) -> torch.Tensor:
        return flash_attn_with_kvcache(
            query,
            keys.view(-1, 1, num_kv_heads, head_dim),
            values.view(-1, 1, num_kv_heads, head_dim),
            page_table=page_table,
            cache_seqlens=window_lengths,
            softmax_scale=scale,
            causal=True,
            window_size=(retained, 0),
            ver=version,
        )

    gathered_keys = keys[addresses.window_rows].view(
        batch_size, ring_frames, num_kv_heads, head_dim
    )
    gathered_values = values[addresses.window_rows].view(
        batch_size, ring_frames, num_kv_heads, head_dim
    )
    frame_offsets = torch.arange(length, device="cuda")
    query_offsets = (addresses.window_lengths - length)[:, None] + frame_offsets
    distances = query_offsets[:, :, None] - torch.arange(ring_frames, device="cuda")
    visible = ((distances >= 0) & (distances < sliding_window))[:, None]
    groups = num_heads // num_kv_heads
    scores = (
        query.float().transpose(1, 2)
        @ gathered_keys.float()
        .repeat_interleave(groups, dim=2)
        .transpose(1, 2)
        .transpose(2, 3)
        * scale
    )
    weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
    reference = (
        weights
        @ gathered_values.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    ).transpose(1, 2)
    fa3, fa3_seconds = timed_first_call(lambda: paged(3))
    fa4, fa4_seconds = timed_first_call(lambda: paged(4))
    fa4_repeat = paged(4)
    graph = torch.cuda.CUDAGraph()
    for _ in range(2):
        paged(4)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        static_output = paged(4)
    query.copy_(
        torch.randn(query.shape, generator=generator, device="cuda").to(torch.bfloat16)
    )
    graph.replay()
    replayed = static_output.clone()
    eager = paged(4)
    return {
        "site": "code2wav_ring",
        "batch_size": batch_size,
        "new_frames": length,
        "ring_frames": ring_frames,
        "window_lengths": window_lengths.tolist(),
        "fa3_rel_l2_fp32": relative_l2(fa3, reference),
        "fa4_rel_l2_fp32": relative_l2(fa4, reference),
        "fa4_vs_fa3_rel_l2": relative_l2(fa4, fa3),
        "fa4_repeat_bit_identical": bool(torch.equal(fa4, fa4_repeat)),
        "fa4_replay_equals_eager": bool(torch.equal(replayed, eager)),
        "fa3_first_call_s": round(fa3_seconds, 4),
        "fa4_first_call_s": round(fa4_seconds, 4),
    }


def main() -> None:
    config_path, out_path = map(pathlib.Path, sys.argv[1:3])
    config = json.loads(config_path.read_text())
    vision = config["thinker_config"]["vision_config"]
    audio = config["thinker_config"]["audio_config"]
    code2wav = config["code2wav_config"]
    generator = torch.Generator(device="cuda").manual_seed(20261027)
    window_input = torch.tensor([audio["n_window"] * 2])
    chunk_tokens = int(
        hf_modeling._get_feat_extract_output_lengths(  # noqa: leading-underscore  # Transformers helper
            window_input, audio["n_window"]
        ).item()
    )
    audio_window = chunk_tokens * (audio["n_window_infer"] // (audio["n_window"] * 2))
    audio_head_dim = audio["d_model"] // audio["encoder_attention_heads"]
    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "packages": {
            name: package_version(name)
            for name in ("sglang", "flash-attn-4", "nvidia-cutlass-dsl", "sgl-kernel")
        },
        "fa4_available": is_flash_attention_v4_available(),
        "audio_window": audio_window,
    }
    rows: list[dict[str, object]] = []
    rows += encoder_cases(
        "vision",
        vision["num_heads"],
        vision["hidden_size"] // vision["num_heads"],
        VISION_SEGMENTS,
        generator,
    )
    rows += encoder_cases(
        "audio",
        audio["encoder_attention_heads"],
        audio_head_dim,
        (
            (audio_window,) * 4 + (audio_window // 3,),
            (audio_window, 0, 0, audio_window // 2, 0),
        ),
        generator,
    )
    rows.append(
        audio_graph_case(
            audio["encoder_attention_heads"], audio_head_dim, audio_window, generator
        )
    )
    code2wav_heads = code2wav["num_attention_heads"]
    for batch_size in CODE2WAV_BATCH_SIZES:
        rows.append(
            code2wav_ring_case(
                code2wav_heads,
                code2wav["num_key_value_heads"],
                code2wav["hidden_size"] // code2wav_heads,
                code2wav["sliding_window"],
                batch_size,
                generator,
            )
        )
    report["rows"] = rows
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
