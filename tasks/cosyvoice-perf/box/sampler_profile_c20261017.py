"""SGLang's sampler on Fun-CosyVoice3's unseeded decode draw, PyTorch backend against FlashInfer: CUDA kernels and
device time per call, and host time per call, at 1, 16 and 32 rows. The draw CosyVoice requests reach: float32 logits
over 6,761 ids, temperature 0.7, top-p 0.8, top-k 20, no seed.
usage (in the container, one card): python3 sampler_profile_c20261017.py"""

import time
from types import SimpleNamespace

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from sglang_omni.model_runner.sglang_model_runner import register_flashinfer_seeded_sampler

VOCAB_SIZE = 6761
CALLS = 200


def sampling_info(rows):
    return SamplingBatchInfo(
        temperatures=torch.full((rows, 1), 0.7, device="cuda"),
        top_ps=torch.full((rows,), 0.8, device="cuda"),
        top_ks=torch.full((rows,), 20, dtype=torch.int32, device="cuda"),
        min_ps=torch.zeros(rows, device="cuda"),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=True,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        sampling_seed=None,
        device="cuda",
    )


def measure(backend, rows):
    generator = torch.Generator(device="cpu").manual_seed(0)
    logits = (torch.randn(rows, VOCAB_SIZE, generator=generator) * 4).to("cuda")
    info = sampling_info(rows)
    positions = torch.arange(rows, device="cuda")
    with (
        get_context().override_server_args(sampling_backend=backend),
        get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
    ):
        sampler = create_sampler()

        def call(step_logits):
            sampler(LogitsProcessorOutput(next_token_logits=step_logits), info, False, [0] * rows,
                    [[] for _ in range(rows)], positions)

        def fresh_logits():
            # the sampler divides and softmaxes the logits in place, so every call gets its own copy, made untimed
            copies = [logits.clone() for _ in range(CALLS)]
            torch.cuda.synchronize()
            return copies

        for step_logits in fresh_logits()[:20]:
            call(step_logits)
        copies = fresh_logits()
        start = time.perf_counter()
        for step_logits in copies:
            call(step_logits)
        torch.cuda.synchronize()
        host_us = 1e6 * (time.perf_counter() - start) / CALLS
        copies = fresh_logits()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for step_logits in copies:
                call(step_logits)
            torch.cuda.synchronize()
    kernels = [event for event in prof.events() if event.device_type == DeviceType.CUDA]
    copy_kernels = [event for event in kernels if "copy" in event.name.lower() or "memcpy" in event.name.lower()]
    device_us = sum(event.device_time for event in kernels) / CALLS
    return len(kernels) / CALLS, len(copy_kernels) / CALLS, device_us, host_us


register_flashinfer_seeded_sampler()
print("| rows | backend | CUDA kernels a call | of them copies | device time a call (us) | wall a call (us) |")
print("| --- | --- | --- | --- | --- | --- |")
for rows in (1, 16, 32):
    for backend in ("pytorch", "flashinfer"):
        kernels, copies, device_us, host_us = measure(backend, rows)
        print(f"| {rows} | {backend} | {kernels:.1f} | {copies:.1f} | {device_us:.1f} | {host_us:.1f} |")
