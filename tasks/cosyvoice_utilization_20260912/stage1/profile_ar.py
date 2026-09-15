#!/usr/bin/env python3
"""Stage 1, AR: one scheduler iteration of the Fun-CosyVoice3 tts_engine at
fixed batch and length, with each GPU activity attributed to the Python range
that launched it.

Run alone on the GPU, in its own process (one SGLang engine per process), with
PYTHONPATH holding the tree under test, stage0 and stage1 (see README.md):

  python profile_ar.py --device cuda:0 --out stage1-out

Steps run in the grad mode the serving scheduler thread has (default, with the
model forward under no_grad), so host dispatch cost matches serving.

The engine is built by the serving factory with the serving factory arguments
(create_sglang_tts_engine_executor, bf16, 16 ONNX threads, hop 25) and never
started. A step is the body of OmniScheduler._event_loop_normal without the
inbox poll: get_next_batch_to_run, run_batch, process_batch_result. So a step
pays everything a served step pays: batch preparation, ForwardBatch build,
the forward (eager prefill from prompt embeddings, decode graph replay),
sampling with the CosyVoice3 penalties, the host token reads, stream chunk
emission and finish handling. Requests enter through the real ingress
(preprocess_cosyvoice3_payload, then process_input_requests with the stage
request builder) from real SeedTTS samples, stream on.

Points, in the spirit of sglang.benchmark.one_batch:
  prefill, one request at the shortest, median and longest prompt of the
    first 400 SeedTTS en samples, 16 requests near the median, and 32 requests
    near the median, of which one step admits what max_prefill_tokens allows;
    each repeat flushes the radix cache so no prefix is reused, and requests
    carry max_new_tokens 1 so the step finishes them and leaves the engine idle
  decode, 1, 16 and 32 running requests: one step with graph replay, one step
    eager (decode_cuda_graph_runner cleared, the path disable_cuda_graph
    takes), and one hop of 25 steps (one stream chunk per request); at 32
    requests also one step after 1,000 generated tokens; stop tokens are held
    off with min_new_tokens equal to max_new_tokens
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics

from common import MODEL_ID, provenance
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.model_runner import ModelRunner as SGLangModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from trace_ledger import measure, render_markdown

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.stages import create_sglang_tts_engine_executor
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

SAMPLE_COUNT = 400
HOP_STEPS = 25
DECODE_ROWS = (1, 16, 32)
LATE_GENERATED = 1000
DECODE_MAX_NEW_TOKENS = LATE_GENERATED + 100


def ar_functions():
    return [
        (omni_scheduler.OmniScheduler, "get_next_batch_to_run"),
        (omni_scheduler.OmniScheduler, "run_batch"),
        (omni_scheduler.OmniScheduler, "process_batch_result"),
        (ScheduleBatch, "prepare_for_extend"),
        (ScheduleBatch, "prepare_for_decode"),
        (ModelRunner, "_build_forward_batch"),
        (ModelRunner, "_prepare_and_forward"),
        (ModelRunner, "_ensure_next_token_ids"),
        (ModelRunner, "_publish_next_tokens"),
        (ModelRunner, "_finalize"),
        (FunCosyVoice3ModelRunner, "_build_prefill_input_embeds"),
        (FunCosyVoice3ModelRunner, "_forward_with_input_embeds"),
        (FunCosyVoice3ModelRunner, "_sample_next_token_ids"),
        (FunCosyVoice3ModelRunner, "_collect_tokens"),
        (FunCosyVoice3ModelRunner, "_emit_code_chunk"),
        (TpModelWorker, "forward_batch_generation"),
        (SGLangModelRunner, "sample"),
        (DecodeCudaGraphRunner, "execute"),
        (SGLangOutputProcessor, "process"),
    ]


class Engine:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.sglang_runner = scheduler._model_runner.tp_worker.model_runner
        self.issued = 0
        self.recording = False
        self.shapes: list[tuple] = []
        self.messages = {"stream": 0, "result": 0}

    def payload(self, sample, max_new_tokens: int) -> StagePayload:
        self.issued += 1
        return StagePayload(
            request_id=f"ar-{self.issued}",
            request=OmniRequest(
                inputs={
                    "text": sample.target_text,
                    "ref_audio": sample.ref_audio,
                    "ref_text": sample.ref_text,
                },
                params={"stream": True, "max_new_tokens": max_new_tokens},
            ),
            data={},
        )

    def prompt_length(self, sample) -> int:
        prepared = request_builders.preprocess_cosyvoice3_payload(
            self.payload(sample, 1)
        )
        return len(self.scheduler._request_builder(prepared).req.origin_input_ids)

    def submit(self, samples, *, max_new_tokens: int, hold: bool) -> None:
        payloads = [
            request_builders.preprocess_cosyvoice3_payload(
                self.payload(sample, max_new_tokens)
            )
            for sample in samples
        ]
        self.scheduler.process_input_requests(payloads)
        if hold:
            for req in self.scheduler.waiting_queue:
                req.sampling_params.min_new_tokens = req.sampling_params.max_new_tokens

    def step(self):
        scheduler = self.scheduler
        batch = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = batch
        if batch:
            rows = len(batch.reqs)
            if self.recording:
                extend = batch.forward_mode.is_extend()
                self.shapes.append(
                    (
                        "extend" if extend else "decode",
                        rows,
                        int(batch.extend_num_tokens) if extend else rows,
                        max(
                            len(r.origin_input_ids) + len(r.output_ids)
                            for r in batch.reqs
                        ),
                    )
                )
            result = scheduler.run_batch(batch)
            if result is omni_scheduler._FAILED_BATCH_RESULT:
                raise RuntimeError("scheduler step failed; see the log above")
            scheduler.process_batch_result(batch, result)
        scheduler.last_batch = batch
        return batch

    def drain(self) -> None:
        while True:
            try:
                message = self.scheduler.outbox.get_nowait()
            except queue.Empty:
                return
            if message.type == "error":
                raise RuntimeError(f"{message.request_id}: {message.data}")
            self.messages[message.type] = self.messages.get(message.type, 0) + 1

    def idle(self) -> None:
        scheduler = self.scheduler
        while self.step() is not None or scheduler.waiting_queue:
            pass
        self.drain()
        if not scheduler.flush_cache(empty_cache=False):
            raise RuntimeError("engine not idle after draining")

    def stop_all(self) -> None:
        for req in list(self.scheduler.running_batch.reqs):
            self.scheduler.abort(req.rid)
        self.idle()

    def run(self, label, call, out_dir, repeats, prepare):
        print(f"profiling {label}", flush=True)
        self.shapes = []

        # note(ratish): prepare also steps the engine (draining leftovers), so
        # only steps inside the measured call count toward the shapes.
        def recorded():
            self.recording = True
            try:
                call()
            finally:
                self.recording = False

        ledger = measure(
            recorded,
            label=label,
            out_dir=out_dir,
            roots={"llm": self.scheduler._model_runner.model},
            functions=ar_functions(),
            repeats=repeats,
            prepare=prepare,
        )
        assert self.shapes, f"{label} ran no batch"
        ledger["step_shapes"] = sorted({shape[:3] for shape in self.shapes})
        ledger["max_sequence_tokens"] = max(shape[3] for shape in self.shapes)
        ledger["steps_per_call"] = len(self.shapes) // (repeats + 3)
        return ledger


def nearest_median(ranked, count):
    middle = len(ranked) // 2
    start = max(0, min(middle - count // 2, len(ranked) - count))
    return [sample for _, sample in ranked[start : start + count]]


def profile_prefill(engine, ranked, out_dir, repeats):
    points = {
        f"ar_prefill_rows1_prompt{ranked[0][0]}": [ranked[0][1]],
        f"ar_prefill_rows1_prompt{ranked[len(ranked) // 2][0]}": [
            ranked[len(ranked) // 2][1]
        ],
        f"ar_prefill_rows1_prompt{ranked[-1][0]}": [ranked[-1][1]],
        "ar_prefill_rows16_median": nearest_median(ranked, 16),
        "ar_prefill_admit_of32_median": nearest_median(ranked, 32),
    }
    ledgers = []
    for label, samples in points.items():

        def prepare(samples=samples):
            engine.idle()
            engine.submit(samples, max_new_tokens=1, hold=False)

        ledgers.append(engine.run(label, engine.step, out_dir, repeats, prepare))
    engine.idle()
    return ledgers


def profile_decode(engine, ranked, out_dir, repeats):
    runner = engine.sglang_runner
    graphs = runner.decode_cuda_graph_runner
    ledgers = []
    for rows in DECODE_ROWS:
        engine.idle()
        engine.submit(
            nearest_median(ranked, rows),
            max_new_tokens=DECODE_MAX_NEW_TOKENS,
            hold=True,
        )
        while True:
            batch = engine.step()
            if batch is not None and batch.forward_mode.is_decode():
                if len(batch.reqs) != rows:
                    raise RuntimeError(
                        f"decode formed {len(batch.reqs)} rows, not {rows}"
                    )
                break
        engine.drain()

        ledgers.append(
            engine.run(
                f"ar_decode_rows{rows}_graph",
                engine.step,
                out_dir,
                repeats,
                engine.drain,
            )
        )
        runner.decode_cuda_graph_runner = None
        try:
            ledgers.append(
                engine.run(
                    f"ar_decode_rows{rows}_eager",
                    engine.step,
                    out_dir,
                    repeats,
                    engine.drain,
                )
            )
        finally:
            runner.decode_cuda_graph_runner = graphs

        def hop():
            for _ in range(HOP_STEPS):
                engine.step()

        ledgers.append(
            engine.run(
                f"ar_decode_hop{HOP_STEPS}_rows{rows}_graph",
                hop,
                out_dir,
                repeats,
                engine.drain,
            )
        )
        if rows == DECODE_ROWS[-1]:
            while (
                min(len(r.output_ids) for r in engine.scheduler.running_batch.reqs)
                < LATE_GENERATED
            ):
                engine.step()
                engine.drain()
            ledgers.append(
                engine.run(
                    f"ar_decode_rows{rows}_graph_generated{LATE_GENERATED}",
                    engine.step,
                    out_dir,
                    repeats,
                    engine.drain,
                )
            )
        engine.stop_all()
    return ledgers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parts", default="prefill,decode")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parts = set(args.parts.split(","))
    os.makedirs(args.out, exist_ok=True)
    traces = os.path.join(args.out, "traces")
    info = provenance(args.device)
    scheduler = create_sglang_tts_engine_executor(
        args.model,
        device=args.device,
        dtype="bfloat16",
        onnx_intra_op_threads=16,
        token_hop_len=25,
    )
    engine = Engine(scheduler)
    runner = engine.sglang_runner
    graphs = runner.decode_cuda_graph_runner
    info.update(
        {
            "sglang_omni": omni_scheduler.__file__,
            "attention_backend": type(runner.attn_backend).__name__,
            "decode_graph_batches": (
                list(graphs.capture_bs) if graphs is not None else []
            ),
            "max_total_num_tokens": int(scheduler.max_total_num_tokens),
            "max_req_len": int(scheduler.max_req_len),
            "max_prefill_tokens": int(scheduler.server_args.max_prefill_tokens),
            "max_running_requests": int(scheduler.server_args.max_running_requests),
        }
    )

    ranked = sorted(
        (engine.prompt_length(sample), index, sample)
        for index, sample in enumerate(
            load_seedtts_samples(
                SEEDTTS_DATASET_ID,
                SAMPLE_COUNT,
                split="en",
                revision=SEEDTTS_DATASET_REVISION,
            )
        )
    )
    ranked = [(length, sample) for length, _, sample in ranked]
    lengths = [length for length, _ in ranked]
    info["prompt_tokens"] = {
        "samples": len(lengths),
        "min": lengths[0],
        "p50": lengths[len(lengths) // 2],
        "mean": statistics.fmean(lengths),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
    }
    print(f"prompt tokens {info['prompt_tokens']}", flush=True)

    ledgers = []
    if "prefill" in parts:
        ledgers += profile_prefill(engine, ranked, traces, args.repeats)
    if "decode" in parts:
        ledgers += profile_decode(engine, ranked, traces, args.repeats)
    info["outbox_messages"] = engine.messages

    with open(os.path.join(args.out, "ar.json"), "w") as handle:
        json.dump({"provenance": info, "ledgers": ledgers}, handle, indent=1)
    markdown = "| point | step shapes (mode, rows, tokens) | steps per call | max sequence tokens |\n|---|---|---|---|\n"
    for ledger in ledgers:
        markdown += (
            f"| {ledger['label']} | {ledger['step_shapes']} | {ledger['steps_per_call']} "
            f"| {ledger['max_sequence_tokens']} |\n"
        )
    markdown += "\n" + render_markdown(ledgers)
    with open(os.path.join(args.out, "ar.md"), "w") as handle:
        handle.write(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
