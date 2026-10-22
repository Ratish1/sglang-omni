"""Compare trained slot convolutions and their graph replay at PR 2999."""

import copy
import json
import statistics
from pathlib import Path

import torch

import review2999_baseline as baseline
from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniIncrementalCodec
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.models.qwen3_omni.components.incremental_codec import (
    SlotDecode,
    slot_conv_stack,
)
from sglang_omni.platforms import current_platform
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder


def relative_error(actual, reference):
    return float((actual.float() - reference.float()).norm() / reference.float().norm())


def make_pool(codec, rows, dtype, capture_slots=0):
    return CodecStatePool(
        codec.spec,
        num_slots=rows,
        capture_slots=capture_slots,
        ring_frames=codec.spec.retained_context + 10,
        device=torch.device("cuda", 0),
        dtype=dtype,
    )


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(2999)
    torch.backends.cudnn.allow_tf32 = False
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    reference = load_code2wav_model(model_id, device="cuda:0", dtype="fp32")
    report = {"model": model_id, "numerics": [], "timings": []}
    for dtype in (torch.bfloat16, torch.float16):
        model = copy.deepcopy(reference).to(dtype)
        model.use_fused_transformer(
            current_platform.get_joint_rope_inplace_kernel(), max_positions=1024
        )
        fuse_vocoder_decoder(model.decoder)
        codec = Qwen3OmniIncrementalCodec(model)
        truth_pool = make_pool(codec, 2, torch.float32)
        old_pool = make_pool(codec, 2, dtype)
        new_pool = make_pool(codec, 2, dtype)
        slots = torch.tensor([1, 0], device="cuda")
        for step, frames in enumerate((1, 10, 1, 7)):
            hidden = torch.randn(2, frames, model.config.hidden_size, device="cuda")
            truth = baseline.slot_conv_stack(
                reference,
                hidden,
                baseline.SlotDecode(
                    pool=truth_pool, slot_index=slots, is_cold=step == 0
                ),
            )
            old = baseline.slot_conv_stack(
                model,
                hidden.to(dtype),
                baseline.SlotDecode(pool=old_pool, slot_index=slots, is_cold=step == 0),
            )
            actual = slot_conv_stack(
                model,
                hidden.to(dtype),
                SlotDecode(pool=new_pool, slot_index=slots, is_cold=step == 0),
                codec.snake_coefficients,
            )
            assert torch.isfinite(actual).all()
            assert actual.shape == truth.shape
            report["numerics"].append(
                dict(
                    dtype=str(dtype),
                    step=step,
                    fresh_frames=frames,
                    baseline_error=relative_error(old, truth),
                    head_error=relative_error(actual, truth),
                    head_baseline_error=relative_error(actual, old),
                )
            )
        for batch in (1, 8):
            hidden = torch.randn(
                batch, 10, model.config.hidden_size, device="cuda", dtype=dtype
            )
            slots = torch.arange(batch, device="cuda")
            for cold in (True, False):
                functions = {}
                for route in ("head", "baseline"):
                    pool = make_pool(codec, batch, dtype)
                    if route == "head":
                        function = lambda h=hidden, p=pool, c=cold: slot_conv_stack(
                            model,
                            h,
                            SlotDecode(pool=p, slot_index=slots, is_cold=c),
                            codec.snake_coefficients,
                        )
                    else:
                        function = (
                            lambda h=hidden, p=pool, c=cold: baseline.slot_conv_stack(
                                model,
                                h,
                                baseline.SlotDecode(
                                    pool=p, slot_index=slots, is_cold=c
                                ),
                            )
                        )
                    for _ in range(3):
                        function()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = function()
                    functions[route] = (graph, output, pool)
                timings = {route: [] for route in functions}
                for iteration in range(6):
                    routes = tuple(functions)
                    for route in routes if iteration % 2 == 0 else routes[::-1]:
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            functions[route][0].replay()
                        end.record()
                        end.synchronize()
                        timings[route].append(start.elapsed_time(end) / 20)
                report["timings"].append(
                    dict(
                        dtype=str(dtype),
                        rows=batch,
                        cold=cold,
                        median_ms={
                            route: statistics.median(values)
                            for route, values in timings.items()
                        },
                    )
                )
        if dtype == torch.bfloat16:
            pool = make_pool(codec, 2, dtype, capture_slots=2)
            runner = Code2WavCudaGraphRunner.build(
                model,
                device=torch.device("cuda", 0),
                num_quantizers=model.config.num_quantizers,
                total_gpu_memory_fraction=1.0,
                graph_keys=tuple(
                    GraphKey(rows, frames, mode)
                    for rows in (1, 2)
                    for mode, frames in (("first", 10), ("next", 10), ("next", 1))
                ),
                model_footprint_bytes=0,
                decode_stream=None,
                incremental_codec=codec,
                codec_pool=pool,
            )
            assert runner.enabled, runner.disable_reason
            eager_pool = make_pool(codec, 2, dtype)
            slots = [pool.acquire(), pool.acquire()]
            staged_slots = torch.tensor(slots, device="cuda")
            for step, frames in enumerate((10, 10, 1)):
                codes = torch.randint(
                    model.config.codebook_size,
                    (2, model.config.num_quantizers, frames),
                    device="cuda",
                )
                expected = codec.decode_slots(
                    codes, eager_pool, staged_slots, first_decode=step == 0
                )
                result = runner.run_slots(codes, slots, first_decode=step == 0)
                assert result.execution_mode == "cuda_graph"
                assert torch.equal(result.output, expected)
                for name in pool.histories:
                    assert torch.equal(
                        pool.histories[name][:2], eager_pool.histories[name]
                    )
                for name in pool.overlaps:
                    assert torch.equal(
                        pool.overlaps[name][:2], eager_pool.overlaps[name]
                    )
                assert torch.equal(pool.frame_positions[:2], eager_pool.frame_positions)
            report["trained_full_graph_replays"] = 3
        print(json.dumps(report), flush=True)
    Path("/data/review2999/checkpoint.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
