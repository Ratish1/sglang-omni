"""Check trained code2wav parity, graph batching and stream cleanup on CUDA."""

from __future__ import annotations

import json
import math
import statistics
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.models.qwen3_omni.components.code2wav import (
    Qwen3OmniCode2Wav,
    Qwen3OmniIncrementalCodec,
)
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
    incremental_graph_keys,
)
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.models.weight_loader import read_safetensors_keys
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder
from tests.unit_test.fixtures.qwen_fakes import make_qwen_payload

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OUTPUT = Path("/data/review2909/checkpoint.json")


def make_pool(codec: Qwen3OmniIncrementalCodec, slots: int) -> CodecStatePool:
    return CodecStatePool(
        codec.spec,
        num_slots=slots,
        capture_slots=8,
        ring_frames=codec.spec.retained_context + 10,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
    )


def parts_forward(
    codec: Qwen3OmniIncrementalCodec,
    codes: torch.Tensor,
    parts: list[int],
    pool: CodecStatePool,
    slots: list[int],
) -> torch.Tensor:
    outputs = []
    position = 0
    for frames in parts:
        output = codec.decode_slots(
            codes[..., position : position + frames],
            pool,
            pool.stage_index(slots),
            first_decode=position == 0,
        )
        outputs.append(output)
        position += frames
    return torch.cat(outputs, dim=-1)


def signal_to_noise(actual: torch.Tensor, reference: torch.Tensor) -> float:
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    error = (actual.float() - reference.float()).square().sum()
    return 10 * math.log10(float(reference.float().square().sum() / error))


def main() -> None:
    torch.manual_seed(2909)
    configuration = AutoConfig.from_pretrained(MODEL)
    index_path = hf_hub_download(MODEL, "model.safetensors.index.json")
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    weights = {}
    for shard in sorted(
        {value for key, value in weight_map.items() if key.startswith("code2wav.")}
    ):
        shard_path = Path(hf_hub_download(MODEL, shard))
        names = [
            key
            for key, value in weight_map.items()
            if value == shard and key.startswith("code2wav.")
        ]
        weights.update(
            {
                key.removeprefix("code2wav."): value
                for key, value in read_safetensors_keys(shard_path, names).items()
            }
        )
    reference = Qwen3OmniMoeCode2Wav._from_config(
        deepcopy(configuration.code2wav_config)
    ).eval()
    reference.load_state_dict(weights, strict=True)
    reference.to(device="cuda", dtype=torch.float32)
    model = Qwen3OmniCode2Wav._from_config(configuration.code2wav_config).eval()
    model.load_state_dict(weights, strict=True)
    model.to(device="cuda", dtype=torch.bfloat16)
    model.use_channels_last()
    model.use_fused_transformer(
        current_platform.get_joint_rope_inplace_kernel(),
        max_positions=configuration.talker_config.text_config.max_position_embeddings,
    )
    fuse_vocoder_decoder(model.decoder)
    codec = Qwen3OmniIncrementalCodec(model)
    pool = make_pool(codec, 32)
    rows = []
    with torch.inference_mode():
        for frames in (11, 71, 72, 81, 82, 171, 290):
            codes = torch.randint(
                model.config.codebook_size,
                (1, model.config.num_quantizers, frames),
                device="cuda",
            )
            wanted = reference(codes)
            whole = model(codes)
            uniform = [10] * (frames // 10) + ([frames % 10] if frames % 10 else [])
            old_windows = []
            position = 0
            for count in uniform:
                context = min(25, position)
                old_windows.append(
                    model(codes[..., position - context : position + count])[
                        ..., -count * model.total_upsample :
                    ]
                )
                position += count
            old_window_snr = signal_to_noise(torch.cat(old_windows, dim=-1), wanted)
            irregular = []
            left = frames
            while left:
                count = min(left, (1, 9, 3, 7)[len(irregular) % 4])
                irregular.append(count)
                left -= count
            for parts in (uniform, irregular):
                actual = parts_forward(codec, codes, parts, pool, [0])
                snr = signal_to_noise(actual, wanted)
                whole_snr = signal_to_noise(whole, wanted)
                assert snr >= whole_snr - 3, (frames, snr, whole_snr)
                row = {
                    "frames": frames,
                    "first_part": parts[0],
                    "parts": len(parts),
                    "snr_db": snr,
                    "whole_snr_db": whole_snr,
                    "old_window_snr_db": old_window_snr,
                    "samples": actual.numel(),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        old_codes = torch.randint(
            model.config.codebook_size,
            (1, model.config.num_quantizers, 290),
            device="cuda",
        )
        parts_forward(codec, old_codes, [10] * 29, pool, [0])
        new_codes = torch.randint(
            model.config.codebook_size,
            (1, model.config.num_quantizers, 29),
            device="cuda",
        )
        reused = parts_forward(codec, new_codes, [1, 9, 10, 9], pool, [0])
        fresh = parts_forward(codec, new_codes, [1, 9, 10, 9], make_pool(codec, 1), [0])
        assert torch.equal(reused, fresh)
        print("trained dirty-slot reuse: bit identical", flush=True)
    runner = Code2WavCudaGraphRunner.build(
        model,
        device=torch.device("cuda:0"),
        num_quantizers=model.config.num_quantizers,
        total_gpu_memory_fraction=0.02,
        graph_keys=(GraphKey(1, 10),) + incremental_graph_keys(10, 8),
        best_effort_keys=tuple(GraphKey(1, frames, "next") for frames in range(1, 10)),
        model_footprint_bytes=sum(tensor.nbytes for tensor in model.parameters())
        + pool.bytes_per_slot * 40,
        decode_stream=None,
        incremental_codec=codec,
        codec_pool=pool,
    )
    assert runner.enabled, runner.stats()
    scheduler = Code2WavScheduler(
        model,
        device="cuda:0",
        stream_chunk_size=10,
        left_context_size=25,
        max_replay_rows=8,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
        incremental_codec=codec,
        codec_pool=pool,
    )
    codes = torch.randint(
        model.config.codebook_size, (8, model.config.num_quantizers, 93), device="cuda"
    )
    request_ids = [f"trained-{row}" for row in range(8)]
    for request_id in request_ids:
        scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    for position in range(0, 90, 10):
        for request_id, stream_item in [
            (
                request_id,
                StreamItem(
                    position,
                    row_codes[:, position : position + 10].T,
                    "talker",
                    metadata={"stream": False},
                ),
            )
            for request_id, row_codes in zip(request_ids, codes)
        ]:
            scheduler.handle_stream_chunk(request_id, stream_item)
        assert len(scheduler.ready_window_group()) == 8
        scheduler.run_ready_step()
    for request_id, row_codes in zip(request_ids, codes):
        scheduler.handle_stream_chunk(
            request_id,
            StreamItem(90, row_codes[:, 90:].T, "talker", metadata={"stream": False}),
        )
        scheduler.handle_stream_done(request_id)
    messages = [scheduler.outbox.get_nowait() for _ in request_ids]
    with torch.inference_mode():
        expected = parts_forward(
            codec, codes, [10] * 9 + [3], make_pool(codec, 8), list(range(8))
        )
    for message, waveform in zip(messages, expected.float().cpu().reshape(8, -1)):
        assert message.type == "result", message
        actual = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
        np.testing.assert_allclose(actual, waveform.numpy(), atol=0.003, rtol=0.03)
    assert pool.available_size() == 32
    print(
        "trained eight-row scheduler graph batching, wrap and final tail: passed",
        flush=True,
    )
    timings = []
    for row_count in (1, 8):
        static_codes = torch.randint(
            model.config.codebook_size,
            (row_count, model.config.num_quantizers, 35),
            device="cuda",
        )
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                model(static_codes)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        old_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(old_graph), torch.inference_mode():
            old_output = model(static_codes)
        assert old_output.numel() > 0
        runner.run_slots(
            static_codes[..., :10], list(range(row_count)), first_decode=True
        )
        for name, graph in (
            ("stateless_35_frames", old_graph),
            (
                "incremental_10_frames",
                runner.graphs[GraphKey(row_count, 10, "next")].graph,
            ),
        ):
            samples = []
            for _ in range(25):
                start_event, end_event = torch.cuda.Event(
                    enable_timing=True
                ), torch.cuda.Event(enable_timing=True)
                start_event.record()
                graph.replay()
                end_event.record()
                end_event.synchronize()
                samples.append(start_event.elapsed_time(end_event))
            timing = {
                "rows": row_count,
                "mode": name,
                "median_ms": statistics.median(samples[5:]),
            }
            timings.append(timing)
            print(json.dumps(timing), flush=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "model": MODEL,
                "rows": rows,
                "bytes_per_slot": pool.bytes_per_slot,
                "graph_stats": runner.stats(),
                "reuse_bit_identical": True,
                "batched_scheduler_rows": 8,
                "timings": timings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
