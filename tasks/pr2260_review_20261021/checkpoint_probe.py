"""Measure trained buffered Flow routes across graph boundaries and dirty replays."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    FunCosyVoice3Flow,
    create_vocoder_executor,
)


def make_inputs(
    token_lengths: tuple[int, ...], prompt_lengths: tuple[int, ...]
) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(sum(token_lengths) + sum(prompt_lengths))
    return [
        FlowBatchInput(
            token=torch.randint(1, 6500, (1, tokens), generator=generator).int(),
            prompt_token=torch.randint(
                1, 6500, (1, prompts), generator=generator
            ).int(),
            prompt_feat=torch.randn(1, prompts * 2, 80, generator=generator),
            embedding=torch.randn(1, 192, generator=generator),
        )
        for tokens, prompts in zip(token_lengths, prompt_lengths, strict=True)
    ]


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double())
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(2260)
    scheduler = create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=3,
        flow_batch_admission_frames=256,
        flow_prefix_cache_gb=0.0,
        enable_dit_torch_compile=False,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=True,
        enable_hift_cuda_graph=False,
    )
    flow = scheduler.vocoder.flow
    assert flow.is_buffered_packed
    runner = flow.whole_history_cuda_graph_runner
    assert runner is not None
    reference = FunCosyVoice3Flow(copy.deepcopy(flow.flow).float())
    cases = [
        ((1, 25, 40), (0, 7, 3)),
        ((20, 37, 52), (1, 3, 15)),
        ((129,), (0,)),
        ((2, 3, 4, 5), (0, 1, 2, 3)),
        ((1, 25, 40), (0, 7, 3)),
    ]
    records = []
    retained_first: list[torch.Tensor] | None = None
    with torch.inference_mode(), scheduler.vocoder.stream_context:
        for token_lengths, prompt_lengths in cases:
            inputs = make_inputs(token_lengths, prompt_lengths)
            graph_hits: list[bool] = []
            real_run = runner.run

            def observe_run(
                *,
                noise: torch.Tensor,
                time_span: torch.Tensor,
                mu: torch.Tensor,
                speaker_embeddings: torch.Tensor,
                mel_conditioning: torch.Tensor,
                lengths: Sequence[int],
            ) -> torch.Tensor | None:
                output = real_run(
                    noise=noise,
                    time_span=time_span,
                    mu=mu,
                    speaker_embeddings=speaker_embeddings,
                    mel_conditioning=mel_conditioning,
                    lengths=lengths,
                )
                graph_hits.append(output is not None)
                return output

            with (
                torch.autocast("cuda", dtype=torch.bfloat16),
                patch.object(runner, "run", observe_run),
            ):
                graph_outputs = flow.inference(inputs)
            frames = 2 * (sum(token_lengths) + sum(prompt_lengths))
            expected_hit = frames <= runner.tier_frames[-1] and len(inputs) <= 3
            assert graph_hits == [expected_hit]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                flow.whole_history_cuda_graph_runner = None
                eager_outputs = flow.inference(inputs)
                flow.whole_history_cuda_graph_runner = runner
                flow.is_buffered_packed = False
                native_outputs = flow.inference(inputs)
                flow.is_buffered_packed = True
            fp32_outputs = reference.inference(inputs)
            row_errors = []
            for tokens, graph, eager, native, fp32 in zip(
                token_lengths,
                graph_outputs,
                eager_outputs,
                native_outputs,
                fp32_outputs,
                strict=True,
            ):
                assert graph.shape == eager.shape == native.shape == fp32.shape
                assert graph.shape == (1, 80, tokens * 2)
                assert torch.isfinite(graph).all() and torch.isfinite(eager).all()
                row_errors.append(
                    {
                        "graph_eager_bit_identical": torch.equal(graph, eager),
                        "graph_eager_relative_error": relative_error(graph, eager),
                        "native_fp32_relative_error": relative_error(native, fp32),
                        "packed_fp32_relative_error": relative_error(graph, fp32),
                    }
                )
            if retained_first is None:
                retained_first = [mel.clone() for mel in graph_outputs]
            elif (token_lengths, prompt_lengths) == cases[0]:
                assert all(
                    torch.equal(actual, first)
                    for actual, first in zip(graph_outputs, retained_first, strict=True)
                )
            else:
                pass
            record = {
                "tokens": token_lengths,
                "prompts": prompt_lengths,
                "total_frames": frames,
                "observed_graph_hit": graph_hits[0],
                "rows": row_errors,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    scheduler.stop()
    summary = {"cases": records, "repeat_after_dirty_replay_bit_identical": True}
    Path("/data/review2260/checkpoint.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
