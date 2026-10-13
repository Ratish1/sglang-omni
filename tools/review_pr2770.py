"""Check sampler warmup reuse, output parity and CPU reachability."""

from __future__ import annotations

import inspect
import json
import time

import torch
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils import DynamicGradMode
from torch._dynamo.utils import counters
from transformers import PretrainedConfig

from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel
from sglang_omni.sampling.sampler_compile import (
    SamplerCompileForms,
    compile_sampler_functions,
)


def check_cuda_forms() -> None:
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(20261013)
    for inference_mode in (False, True):
        DynamicGradMode.set_inference_mode(inference_mode)
        for logits_dtype in (torch.bfloat16, torch.float32):
            torch.compiler.reset()
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(device).clone()
            started = time.perf_counter()
            compile_sampler_functions(
                SamplerCompileForms(
                    device=device,
                    vocab_size=3072,
                    seeded_log_probability_dtype=torch.float64,
                    penalized_logits_dtype=logits_dtype,
                )
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
            compiled_frames = counters["frames"]["ok"]
            with DynamicGradMode():
                for rows in (1, 2, 5, 13, 64):
                    logits = torch.randn(
                        rows,
                        3072,
                        device=device,
                        dtype=logits_dtype,
                        generator=generator,
                    )
                    penalties = torch.full_like(logits, 1.21, dtype=torch.float32)
                    expected_logits = logits.clone()
                    inspect.unwrap(apply_scaling_penalties)(expected_logits, penalties)
                    apply_scaling_penalties(logits, penalties)
                    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
                    probabilities = torch.log_softmax(logits.double(), dim=-1)
                    seeds = torch.arange(100, 100 + rows, device=device)
                    positions = torch.arange(20, 20 + rows, device=device)
                    expected_tokens = inspect.unwrap(multinomial_with_seed)(
                        probabilities, seeds, positions
                    )
                    actual_tokens = multinomial_with_seed(
                        probabilities, seeds, positions
                    )
                    assert torch.equal(expected_tokens, actual_tokens)
            assert counters["frames"]["ok"] == compiled_frames
            print(
                json.dumps(
                    {
                        "case": "cuda_forms",
                        "inference_mode": inference_mode,
                        "logits_dtype": str(logits_dtype),
                        "warmup_seconds": elapsed,
                        "post_warmup_compiles": 0,
                        "exact_outputs": True,
                        "rng_and_grad_restored": True,
                    }
                ),
                flush=True,
            )
    DynamicGradMode.set_inference_mode(False)


def check_moss(device: torch.device) -> None:
    model = object.__new__(MossTTSDelaySGLangModel)
    torch.nn.Module.__init__(model)
    model.config = PretrainedConfig(vocab_size_list=[3072])
    model.embedding_list = torch.nn.ModuleList(
        [torch.nn.Embedding(3072, 4, device=device)]
    )
    torch.compiler.reset()
    before_warmup = counters["frames"]["ok"]
    started = time.perf_counter()
    model.precompile_kernels_after_loading()
    elapsed = time.perf_counter() - started
    after_warmup = counters["frames"]["ok"]
    with DynamicGradMode():
        for rows in (1, 3, 16):
            logits = torch.linspace(-3, 3, rows * 3072, device=device).reshape(
                rows, 3072
            )
            sampled = MossTTSModelRunner.sample_tokens(
                logits,
                temperature=0.9,
                top_p=0.95,
                top_k=20,
                seeds=torch.arange(rows, device=device),
                positions=torch.arange(rows, device=device),
            )
            assert sampled.shape == (rows,)
            assert bool(((sampled >= 0) & (sampled < 3072)).all())
    print(
        json.dumps(
            {
                "case": "moss_real_hook_and_consumer",
                "device": str(device),
                "warmup_seconds": elapsed,
                "warmup_frames": after_warmup - before_warmup,
                "consumer_frames": counters["frames"]["ok"] - after_warmup,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    check_cuda_forms()
    check_moss(torch.device("cuda", 0))
    check_moss(torch.device("cpu"))
