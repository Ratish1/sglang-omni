"""Check short-cache attention boundaries and time the actual predictor chain."""

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from torch import nn

from sglang_omni.utils import predictor_layers
from tests.unit_test.fixtures.qwen_predictor import TupleLinear
from tests.unit_test.qwen3_omni import test_predictor_kernels as fixtures


def attention_cases() -> list[dict[str, int | float]]:
    records = []
    for group in (1, 2, 8):
        for batch in (1, 3, 32, 64):
            for tokens, end in ((2, 2), (1, 1), (1, 2), (1, 10), (1, 16), (1, 17)):
                generator = torch.Generator(device="cuda").manual_seed(3033)
                q = torch.randn(
                    batch * tokens, 2048, device="cuda", generator=generator
                ).bfloat16()
                cache_shape = (batch, 17, 16 // group, 128)
                keys = torch.randn(
                    cache_shape, device="cuda", generator=generator
                ).bfloat16()
                values = torch.randn(
                    cache_shape, device="cuda", generator=generator
                ).bfloat16()
                keys[:, end:] = float("nan")
                values[:, end:] = float("nan")
                storage = torch.full(
                    (batch * tokens + 8, 2048), 7, device="cuda", dtype=torch.bfloat16
                )
                output = storage[: batch * tokens]
                shape = predictor_layers.PredictorLayerShape(
                    1024, 128, 16, 16 // group, 3072, 1, 1
                )

                def run() -> None:
                    predictor_layers.decode_attention(
                        q=q,
                        tokens_per_row=tokens,
                        k_cache=keys,
                        v_cache=values,
                        end=end,
                        attention_output=output,
                        shape=shape,
                    )

                run()
                actual = output.clone()
                assert torch.isfinite(actual).all()
                assert torch.all(storage[batch * tokens :] == 7)
                queries = q.view(batch, tokens, 16, 128).transpose(1, 2)
                operands = (
                    queries,
                    keys[:, :end].transpose(1, 2),
                    values[:, :end].transpose(1, 2),
                )
                expected = torch.nn.functional.scaled_dot_product_attention(
                    *(tensor.float() for tensor in operands),
                    is_causal=tokens == 2,
                    enable_gqa=True,
                )
                native = torch.nn.functional.scaled_dot_product_attention(
                    *operands, is_causal=tokens == 2, enable_gqa=True
                )
                actual_heads = actual.view(batch, tokens, 16, 128).transpose(1, 2)
                error = float(
                    (actual_heads.float() - expected).norm() / expected.norm()
                )
                native_error = float(
                    (native.float() - expected).norm() / expected.norm()
                )
                if tokens == 2:
                    keys[:, 1] *= 100
                    values[:, 1] += 100
                    run()
                    assert torch.equal(
                        output.view(batch, tokens, -1)[:, 0],
                        actual.view(batch, tokens, -1)[:, 0],
                    )
                    assert not torch.equal(
                        output.view(batch, tokens, -1)[:, 1],
                        actual.view(batch, tokens, -1)[:, 1],
                    )
                records.append(
                    {
                        "group": group,
                        "batch": batch,
                        "tokens": tokens,
                        "end": end,
                        "relative_error": error,
                        "sdpa_relative_error": native_error,
                    }
                )
    for group, should_fuse in ((8, True), (16, False)):
        original = fixtures.NUM_KV_HEADS
        fixtures.NUM_KV_HEADS = 16 // group
        try:
            model = fixtures.build_talker(torch.device("cuda"), seed=3, max_bs=1)
            resolved = predictor_layers.resolve_predictor_layer_shape(
                model.code_predictor, 17, torch.device("cuda")
            )
            assert (resolved is not None) == should_fuse
        finally:
            fixtures.NUM_KV_HEADS = original
    return records


def benchmark(batch: int) -> dict[str, int | float | list[float]]:
    device = torch.device("cuda")
    model = fixtures.fuse(fixtures.build_talker(device, seed=3033))
    model.code_predictor.lm_head = [
        TupleLinear(1024, 2048).to(device, torch.bfloat16) for _ in range(15)
    ]
    model.code_predictor.model.codec_embedding = [
        nn.Embedding(2048, 1024).to(device, torch.bfloat16) for _ in range(15)
    ]
    embedding = nn.Embedding(3072, 1024).to(device, torch.bfloat16)
    model.get_input_embeddings = lambda: embedding
    model.config = SimpleNamespace(num_code_groups=16)
    model.predictor_input_buffer = torch.zeros(
        64, 2, 1024, device=device, dtype=torch.bfloat16
    )
    model.output_codes = torch.zeros(64, 16, device=device, dtype=torch.long)
    model.output_embeds = torch.zeros(64, 1024, device=device, dtype=torch.bfloat16)
    codes = torch.zeros(batch, 1, device=device, dtype=torch.long)
    hidden = torch.randn(batch, 1, 1024, device=device, dtype=torch.bfloat16)

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        return model.code_predictor_forward_incremental_eager(codes, hidden)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_codes, captured_embeddings = step()
    for seed in (4, 5):
        torch.manual_seed(seed)
        hidden.normal_()
        codes.random_(0, 3072)
        graph.replay()
        replayed = captured_codes.clone(), captured_embeddings.clone()
        eager = step()
        assert torch.equal(replayed[0], eager[0])
        assert torch.equal(replayed[1], eager[1])
    for _ in range(30):
        graph.replay()
    times = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 50)
    return {"batch": batch, "median_ms": statistics.median(times), "windows_ms": times}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--batch", type=int)
    parsed = parser.parse_args()
    torch.set_num_threads(1)
    with (
        torch.inference_mode(),
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
    ):
        measurements = (
            attention_cases() if parsed.batch is None else benchmark(parsed.batch)
        )
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "production_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD:sglang_omni"], text=True
        ).strip(),
        "measurements": measurements,
    }
    Path(f"/data/review3033/{parsed.label}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
