"""Host and device cost of each decode-step component the async decode slices change, measured directly, no profiler.

usage: PYTHONPATH=<tree> python3 component_costs.py --card 0 [--iterations 300]
Every component calls the real code of the tree on PYTHONPATH (main, S1, S2a or S3, told apart by what the tree
defines), with only the talker forward, SGLang's sampling kernel and the code predictor left out, since no slice
changes them. Batch sizes 1, 16 and 64; the checkpoint's shapes (talker hidden 2048, vocab 3072, 16 codebooks):
  penalty_prepare  SGLang's per-step penalty work before the forward: the last ids gathered on the host and scattered
                   into the per-request state (only while a request asks SGLang for a penalty), then the forward's
                   sampling copy, which accumulates the penalties into fresh tensors
  decode_inputs    Qwen3TTSModelRunner.write_feedback_buffers on a decode batch past its trailing text
  sample_hooks     Qwen3TTSModelRunner.sample_next_token_ids around the sampler: codec suppression, the leading-silence
                   mask, seeds, SGLang's penalty application inside the sampler, and S1's slot penalty
  code_step        post_decode then post_process_outputs: the staged ids, the feedback rows, the frame snapshot and the
                   commit loop
  lookahead_loop   what the lookahead loop adds per step: two ScheduleBatch.copy(), the scheduler output replace, the
                   completion event record and its query
Host time: perf_counter around one call, the device synchronized before it. Device time: CUDA events around one call
enqueued behind a long device sleep, so its kernels run back to back without launch gaps. Medians and the host p90
over the iterations after a warmup.
"""

import argparse
import dataclasses
import statistics
import time
from types import SimpleNamespace

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib import (
    BatchedFrequencyPenalizer,
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedPresencePenalizer,
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

import sglang_omni
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput, SchedulerRequest

HIDDEN = 2048
VOCAB = 3072
CODEBOOKS = 16
CODEC_EOS = 2150
REPETITION_PENALTY = 1.05
PROMPT_LENGTH = 100
TRAILING_TEXT_ROWS = 40
GENERATED_FRAMES = 50
BATCH_SIZES = (1, 16, 64)
SLOT_COUNT = 65
WARMUP = 30
SLEEP_CYCLES = 20_000_000

PENALTY_ON_SLOTS = Qwen3TTSModelRunner.sample_next_token_ids is not ModelRunner.sample_next_token_ids
FEEDBACK_ON_SLOTS = "text_rows" in Qwen3TTSSGLangRequestData.__dataclass_fields__
CODE_STEP_LAUNCHED = hasattr(Qwen3TTSModelRunner, "launch_code_step")


class PenaltyBatch:
    def __init__(self, reqs: list, device: torch.device) -> None:
        self.reqs = reqs
        self.device = device
        self.sampling_info: SamplingBatchInfo | None = None


def implementation() -> str:
    if CODE_STEP_LAUNCHED:
        return "S3"
    elif FEEDBACK_ON_SLOTS:
        return "S2a"
    elif PENALTY_ON_SLOTS:
        return "S1"
    else:
        return "main"


def measure(call, setup, iterations: int) -> tuple[float, float, float]:
    host_us: list[float] = []
    for iteration in range(WARMUP + iterations):
        setup()
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        host = time.perf_counter() - start
        torch.cuda.synchronize()
        if iteration >= WARMUP:
            host_us.append(host * 1e6)
        else:
            pass
    device_us: list[float] = []
    for iteration in range(WARMUP + iterations):
        setup()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(SLEEP_CYCLES)  # noqa: leading-underscore  # upstream name
        start_event.record()
        call()
        end_event.record()
        torch.cuda.synchronize()
        if iteration >= WARMUP:
            device_us.append(start_event.elapsed_time(end_event) * 1e3)
        else:
            pass
    host_us.sort()
    return statistics.median(host_us), host_us[int(0.9 * len(host_us))], statistics.median(device_us)


def make_runner(device: torch.device, sampled_ids: torch.Tensor) -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.device = device
    runner.row_ids_cache = None
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.codes_event_module = torch.cuda
    runner.leading_silence_mask_frames = 2
    runner.silence_codec_ids = torch.tensor([5, 7, 11], device=device)
    embedding = torch.nn.Embedding(64, HIDDEN, device=device, dtype=torch.bfloat16)
    embedding.weight.requires_grad_(False)
    runner.model = SimpleNamespace(
        decode_feedback_embedding=embedding,
        output_codes=torch.zeros(64, CODEBOOKS, dtype=torch.long, device=device),
        output_embeds=torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device=device),
        config=SimpleNamespace(codec_eos_token_id=CODEC_EOS, vocab_size=VOCAB),
        code_predictor_forward=lambda layer0_codes, hidden, semantic_positions=None: None,
    )

    def sample(logits_output, forward_batch):
        forward_batch.sampling_info.apply_logits_bias(logits_output.next_token_logits)
        return sampled_ids[: logits_output.next_token_logits.shape[0]]

    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sample=sample))
    if PENALTY_ON_SLOTS:
        runner.repetition_scales = torch.ones(SLOT_COUNT, VOCAB, device=device)
        runner.repetition_penalties = torch.full((SLOT_COUNT,), REPETITION_PENALTY, device=device)
    else:
        pass
    if FEEDBACK_ON_SLOTS:
        runner.feedback_pool = torch.zeros(SLOT_COUNT, HIDDEN, dtype=torch.bfloat16, device=device)
    else:
        pass
    if CODE_STEP_LAUNCHED:
        runner.resolving_code_step = None
    else:
        runner.has_pending_code_step = False
    return runner


def make_requests(batch_size: int, device: torch.device) -> list[SchedulerRequest]:
    pad_row = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    text_rows = torch.randn(TRAILING_TEXT_ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    requests = []
    for row in range(batch_size):
        sglang_request = SimpleNamespace(
            rid=f"r{row}",
            output_ids=list(range(GENERATED_FRAMES)),
            origin_input_ids=[0] * PROMPT_LENGTH,
            inflight_middle_chunks=0,
            is_retracted=False,
            finished=lambda: False,
            sampling_params=SimpleNamespace(sampling_seed=None),
        )
        data = Qwen3TTSSGLangRequestData(
            req=sglang_request, mask_leading_silence=True, repetition_penalty=REPETITION_PENALTY
        )
        data.output_codes = [torch.zeros(CODEBOOKS, dtype=torch.long, device=device)] * GENERATED_FRAMES
        if FEEDBACK_ON_SLOTS:
            data.text_rows = torch.cat((text_rows, pad_row.reshape(1, -1)))
        else:
            data.tts_pad_embed = pad_row
        requests.append(SchedulerRequest(request_id=f"r{row}", data=data))
    return requests


def make_sampling_info(batch_size: int, device: torch.device) -> tuple[PenaltyBatch, SamplingBatchInfo]:
    sglang_repetition_penalty = 1.0 if PENALTY_ON_SLOTS else REPETITION_PENALTY
    penalty_batch = PenaltyBatch(
        [
            SimpleNamespace(
                output_ids=list(range(GENERATED_FRAMES)),
                origin_input_ids=[0] * PROMPT_LENGTH,
                sampling_params=SimpleNamespace(
                    repetition_penalty=sglang_repetition_penalty,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    min_new_tokens=0,
                ),
            )
            for _ in range(batch_size)
        ],
        device,
    )
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size=VOCAB,
        batch=penalty_batch,
        penalizers={
            BatchedFrequencyPenalizer,
            BatchedMinNewTokensPenalizer,
            BatchedPresencePenalizer,
            BatchedRepetitionPenalizer,
        },
    )
    penalty_batch.sampling_info = SamplingBatchInfo(
        temperatures=torch.full((batch_size, 1), 0.9, device=device),
        top_ps=torch.ones(batch_size, device=device),
        top_ks=torch.full((batch_size,), 50, dtype=torch.int32, device=device),
        min_ps=torch.zeros(batch_size, device=device),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=VOCAB,
        penalizer_orchestrator=orchestrator,
        device=str(device),
    )
    return penalty_batch, penalty_batch.sampling_info


def make_forward_batch(batch_size: int, device: torch.device, sampling_info: SamplingBatchInfo) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
        req_pool_indices=torch.arange(1, batch_size + 1, device=device),
        seq_lens_cpu=torch.full((batch_size,), PROMPT_LENGTH + GENERATED_FRAMES + 1),
        positions=torch.full((batch_size,), PROMPT_LENGTH + GENERATED_FRAMES, device=device),
        forward_mode=SimpleNamespace(is_decode=lambda: True, is_extend=lambda: False),
        sampling_info=sampling_info.copy_for_forward(),
    )


def components(batch_size: int, device: torch.device) -> dict[str, tuple]:
    sampled_ids = torch.randint(0, 2048, (batch_size,), device=device)
    runner = make_runner(device, sampled_ids)
    requests = make_requests(batch_size, device)
    penalty_batch, sampling_info = make_sampling_info(batch_size, device)
    forward_batch = make_forward_batch(batch_size, device, sampling_info)

    def penalty_prepare() -> None:
        if sampling_info.penalizer_orchestrator.is_required:
            ScheduleBatch.cumulate_penalty_output_tokens(penalty_batch)
        else:
            pass
        sampling_info.copy_for_forward()

    feedback_row = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)

    def queue_feedback() -> None:
        if FEEDBACK_ON_SLOTS:
            pass
        else:
            for scheduled_request in requests:
                scheduled_request.data.pending_feedback_queue.append(feedback_row)

    logits = torch.randn(batch_size, VOCAB, device=device)

    def sample_hooks() -> None:
        logits_output = SimpleNamespace(next_token_logits=logits)
        runner.sample_next_token_ids(logits_output, forward_batch, None, requests)

    hidden = torch.randn(batch_size, HIDDEN, dtype=torch.bfloat16, device=device)
    outputs = {
        scheduled_request.request_id: RequestOutput(scheduled_request.request_id, data=7)
        for scheduled_request in requests
    }
    scheduler_output = SchedulerOutput(requests=requests, batch_data=None)

    def code_step() -> None:
        result = SimpleNamespace(next_token_ids=sampled_ids, logits_output=SimpleNamespace(hidden_states=hidden))
        runner.post_decode(result, forward_batch, None, requests)
        runner.post_process_outputs(result, scheduler_output, outputs)

    schedule_batch = ScheduleBatch(reqs=[object() for _ in range(batch_size)])

    def lookahead_loop() -> None:
        resolve_batch = schedule_batch.copy()
        dataclasses.replace(scheduler_output, batch_data=resolve_batch)
        completion = torch.cuda.Event()
        completion.record()
        schedule_batch.copy()
        completion.query()

    return {
        "penalty_prepare": (penalty_prepare, lambda: None),
        "decode_inputs": (lambda: runner.write_feedback_buffers(forward_batch, requests), queue_feedback),
        "sample_hooks": (sample_hooks, lambda: None),
        "code_step": (code_step, lambda: None),
        "lookahead_loop": (lookahead_loop, lambda: None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.card}")
    torch.cuda.set_device(device)
    print(f"implementation {implementation()} from {sglang_omni.__file__}; {args.iterations} iterations after {WARMUP}")
    print("component | bs | host median us | host p90 us | device median us")
    with torch.no_grad():
        for batch_size in BATCH_SIZES:
            for name, (call, setup) in components(batch_size, device).items():
                host_median, host_p90, device_median = measure(call, setup, args.iterations)
                print(f"{name} | {batch_size} | {host_median:.1f} | {host_p90:.1f} | {device_median:.1f}")


if __name__ == "__main__":
    main()
