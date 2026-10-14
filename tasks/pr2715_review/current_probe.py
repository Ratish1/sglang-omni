"""Compare production-factory Flow paths between the review head and its base."""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    create_vocoder_executor,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import causal_hop_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.device)
    torch.manual_seed(2715)
    started = time.monotonic()
    scheduler = create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=arguments.device,
        max_batch_size=4,
        flow_batch_admission_frames=512,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        flow_prefix_cache_gb=1.0,
    )
    startup_seconds = time.monotonic() - started
    vocoder = scheduler.vocoder
    flow = vocoder.flow
    generator = torch.Generator().manual_seed(2715)

    def make_input(tokens: int, prompt_tokens: int) -> FlowBatchInput:
        return FlowBatchInput(
            token=torch.randint(0, 6561, (1, tokens), generator=generator, dtype=torch.int32),
            prompt_token=torch.randint(
                0, 6561, (1, prompt_tokens), generator=generator, dtype=torch.int32
            ),
            prompt_feat=torch.randn(
                1, prompt_tokens * flow.token_mel_ratio, flow.output_size, generator=generator
            ),
            embedding=torch.randn(1, flow.spk_embed_affine_layer.in_features, generator=generator),
        )

    outputs: dict[str, torch.Tensor] = {}
    calls: dict[str, int] = {}
    for name, runner in (
        ("prefix", flow.prefix_cuda_graph_runner),
        ("whole_history", flow.whole_history_cuda_graph_runner),
    ):
        assert runner is not None
        original = runner.run

        def counted(*, _name=name, _original=original, **inputs):
            output = _original(**inputs)
            key = f"{_name}_{'replay' if output is not None else 'miss'}"
            calls[key] = calls.get(key, 0) + 1
            return output

        runner.run = counted

    with torch.inference_mode(), vocoder.stream_context:
        item = make_input(28, 25)
        cache = vocoder.prefix_cache_rows(
            causal_hop_frames(item, token_mel_ratio=flow.token_mel_ratio)
        )
        assert cache is not None
        try:
            for index, extra_tokens in enumerate((0, 25, 100)):
                if extra_tokens:
                    item = FlowBatchInput(
                        token=torch.cat(
                            (
                                item.token,
                                torch.randint(
                                    0, 6561, (1, extra_tokens), generator=generator, dtype=torch.int32
                                ),
                            ),
                            dim=1,
                        ),
                        prompt_token=item.prompt_token,
                        prompt_feat=item.prompt_feat,
                        embedding=item.embedding,
                    )
                    assert vocoder.grow_prefix_cache(
                        cache, causal_hop_frames(item, token_mel_ratio=flow.token_mel_ratio)
                    )
                outputs[f"prefix_{index}"] = vocoder.hop_batch_prefix([item], [cache])[0].cpu()
        finally:
            vocoder.release_prefix_cache(cache)

        for name, token_lengths in (
            ("causal", (28, 53, 128)),
            ("final", (28, 53)),
            ("above_tier", (300,)),
        ):
            inputs = [make_input(length, 25) for length in token_lengths]
            compute = vocoder.hop_batch if name == "causal" else vocoder.leftover_batch
            for index, mel in enumerate(compute(inputs)):
                outputs[f"{name}_{index}"] = mel.cpu()

        timing_input = [make_input(53, 25)]
        timings = []
        for iteration in range(24):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            vocoder.leftover_batch(timing_input)
            end.record()
            end.synchronize()
            if iteration >= 4:
                timings.append(begin.elapsed_time(end))

    assert all(torch.isfinite(mel).all() for mel in outputs.values())
    assert len(outputs) == 9
    assert calls["prefix_replay"] == 3
    assert calls["whole_history_miss"] == 1
    assert flow.prefix_pool.free_frames > 0
    summary = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "startup_seconds": startup_seconds,
        "mel_count": len(outputs),
        "graph_calls": calls,
        "final_solve_median_ms": statistics.median(timings),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(arguments.device),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, arguments.output)
    arguments.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
