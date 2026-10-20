"""Kit trace pair of one served code2wav decode shape, for analyze_llm_torch_profile.py and kit_replay_span_read.py.

Run on the box from a tree (split_kit_queue.py does): PYTHONPATH=<tree>:<tree>/.claude/skills/omni-gpu-deep-dive/scripts
    python3 /workspace/box/tasks/pr2800_review/box/code2wav_kit_slots.py
    --rows 8 --decode next --frames 10 --output-dir /data/pr2800/<run>/<job>
The served bf16 code2wav (load_code2wav_model, fused SnakeBeta) and its graph runner as create_code2wav_scheduler builds
them, on card 0. decode "window" is a stateless window of the given frames (upstream's walk: 10 for a stream's first
window, 35 for a later one) and runs on any tree; "first" and "next" decode the given new frames from the rows' slots
in a codec state pool (Qwen3OmniIncrementalCodec), a stream's first window or a later one. Mapping: the runner's eager
fallback. Formal: the runner's graph replay under no_grad, as the scheduler runs it. capture_pair warms both and
refuses a trace holding a compile or capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omni_trace_pair import capture_pair

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import Code2WavCudaGraphRunner, GraphKey
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import load_code2wav_model
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--decode", choices=("next", "first", "window"), required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    model = load_code2wav_model(MODEL, device=str(device), dtype="bfloat16")
    fuse_vocoder_decoder(model.decoder)
    codes = torch.randint(
        0, int(model.config.codebook_size), (args.rows, int(model.config.num_quantizers), args.frames), device=device
    )
    if args.decode == "window":
        key = GraphKey(batch_size=args.rows, frames=args.frames)
        runner = Code2WavCudaGraphRunner.build(
            model,
            device=device,
            num_quantizers=int(model.config.num_quantizers),
            total_gpu_memory_fraction=1.0,
            graph_keys=(key,),
            model_footprint_bytes=0,
            decode_stream=None,
        )

        def eager_decode() -> torch.Tensor:
            return runner.eager(codes, key=key, reason="mapping").output

        def graph_decode() -> torch.Tensor:
            with torch.no_grad():
                result = runner.run(codes)
            assert result.execution_mode == "cuda_graph", result
            return result.output

    else:
        from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniIncrementalCodec

        # note (ratish): the v2 trees keep the pool in scheduling and read the spec by a getter;
        # the Omni-only 1a (a7290d8c9 on) keeps it in the qwen3_omni components and reads .spec.
        if (Path.cwd() / "sglang_omni/models/qwen3_omni/components/codec_state_pool.py").exists():
            from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
        else:
            from sglang_omni.scheduling.codec_state_pool import CodecStatePool

        codec = Qwen3OmniIncrementalCodec(model)
        spec = codec.spec
        pool = CodecStatePool(
            spec,
            num_slots=args.rows,
            capture_slots=args.rows,
            ring_frames=spec.retained_context + args.frames,
            device=device,
            dtype=model.dtype,
        )
        key = GraphKey(batch_size=args.rows, frames=args.frames, decode=args.decode)
        first_decode = args.decode == "first"
        runner = Code2WavCudaGraphRunner.build(
            model,
            device=device,
            num_quantizers=int(model.config.num_quantizers),
            total_gpu_memory_fraction=1.0,
            graph_keys=(key,),
            model_footprint_bytes=0,
            decode_stream=None,
            incremental_codec=codec,
            codec_pool=pool,
        )
        slots = [pool.acquire() for _ in range(args.rows)]

        def eager_decode() -> torch.Tensor:
            return runner.eager_slots(codes, slots, first_decode=first_decode, key=key, reason="mapping").output

        def graph_decode() -> torch.Tensor:
            with torch.no_grad():
                result = runner.run_slots(codes, slots, first_decode=first_decode)
            assert result.execution_mode == "cuda_graph", result
            return result.output

    assert runner.stats()["build"]["published_graph_count"] == 1, runner.stats()
    # note (ratish): serving stages its first slot ids on the graph path, under no_grad; the pool's staging ring
    # takes that caller's mode, and the eager fallback's inference mode may then write into it.
    graph_decode()
    capture_pair(output_dir=args.output_dir, mapping_body=eager_decode, formal_body=graph_decode, iters=20, warmup=5)


if __name__ == "__main__":
    main()
