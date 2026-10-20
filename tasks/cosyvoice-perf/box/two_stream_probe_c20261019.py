"""S0 of DESIGN_VOCODER_STANDARD_20261019.md: does a first hop's vocoder step on a stream one priority above the
follow-up work finish near its own time while a follow-up step runs, and what does it cost the follow-up step?

Two vocoders built by the serving factory in one process, each with its own graph pool, prefix pool and stream (the
standard's rule that graphs replayed at once never share a pool), on the checkpoint, at the shapes serving runs:

  follow-up steps (vocoder A, default priority): a final step of 4 rows (whole history Flow over a 100 token prompt
  and 200 tokens, HiFT final) and a hop step of 16 rows (a 50 token second hop against each row's cached prefix);
  first hop (vocoder B): one row, a 100 token prompt and 28 tokens, HiFT over its 50 frames.

Per follow-up kind, N rounds of: each step alone; the first hop launched 5 ms into the follow-up step on the
follow-up's own stream (one device queue, today's order on the device), on a second stream at the same priority,
and on a second stream one priority higher. Reported: the first hop's latency from its launch to its result on the
host, the follow-up step's wall, and both together. Each step ends in its host copy, as serving's steps do.

usage (in the container, one card): CUDA_VISIBLE_DEVICES=<card> python3 two_stream_probe_c20261019.py [--rounds N]
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    HiftStepRow,
    create_vocoder_executor,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import causal_hop_frames
from sglang_omni.scheduling.streaming_vocoder import vocoder_decode_stream_priority

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
PROMPT_TOKENS = 100
FIRST_HOP_TOKENS = 28
SECOND_HOP_END_TOKENS = 78
FINAL_TOKENS = 200
FINAL_ROWS = 4
HOP_ROWS = 16
LAUNCH_DELAY_S = 0.005


def build(prefix_gb: float):
    scheduler = create_vocoder_executor(
        MODEL,
        device="cuda",
        gpu_id=0,
        max_batch_size=16,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        enable_hift_cuda_graph=True,
        flow_prefix_cache_gb=prefix_gb,
    )
    return scheduler.vocoder


def flow_input(vocoder, tokens: int) -> FlowBatchInput:
    flow = vocoder.flow
    generator = torch.Generator().manual_seed(tokens)
    return FlowBatchInput(
        token=torch.randint(0, 6561, (1, tokens), dtype=torch.int32, generator=generator),
        prompt_token=torch.randint(0, 6561, (1, PROMPT_TOKENS), dtype=torch.int32, generator=generator),
        prompt_feat=torch.randn(1, PROMPT_TOKENS * flow.token_mel_ratio, flow.output_size, generator=generator),
        embedding=torch.randn(1, flow.spk_embed_affine_layer.in_features, generator=generator),
    )


class FinalStep:
    def __init__(self, vocoder) -> None:
        self.vocoder = vocoder
        self.items = [flow_input(vocoder, FINAL_TOKENS) for _ in range(FINAL_ROWS)]
        frames = FINAL_TOKENS * vocoder.flow.token_mel_ratio
        self.emitted = (frames - 3 * vocoder.flow.token_mel_ratio - vocoder.hift_hold_frames) * (
            vocoder.hift_samples_per_mel_frame
        )

    def __call__(self) -> None:
        mels = self.vocoder.leftover_batch(self.items)
        self.vocoder.hift_step(
            [HiftStepRow(history=mel, emitted_samples=self.emitted, is_final=True) for mel in mels]
        )


class HopStep:
    def __init__(self, vocoder) -> None:
        self.vocoder = vocoder
        ratio = vocoder.flow.token_mel_ratio
        first = [flow_input(vocoder, FIRST_HOP_TOKENS) for _ in range(HOP_ROWS)]
        self.items = [
            FlowBatchInput(
                token=torch.cat([item.token] * 3, dim=1)[:, :SECOND_HOP_END_TOKENS],
                prompt_token=item.prompt_token,
                prompt_feat=item.prompt_feat,
                embedding=item.embedding,
            )
            for item in first
        ]
        self.caches = [vocoder.prefix_cache_rows(causal_hop_frames(item, token_mel_ratio=ratio)) for item in first]
        assert all(cache is not None for cache in self.caches), "prefix pool too small"
        with torch.inference_mode(), vocoder.stream_context:
            first_mels = vocoder.hop_batch_prefix(first, self.caches)
        for cache, item in zip(self.caches, self.items):
            assert vocoder.grow_prefix_cache(cache, causal_hop_frames(item, token_mel_ratio=ratio))
        torch.cuda.synchronize()
        self.saved = [
            [(row.committed_frames, row.conv_context.clone()) for row in pair] for pair in self.caches
        ]
        first_frames = int(first_mels[0].shape[2])
        self.history_frames = (SECOND_HOP_END_TOKENS - 3) * ratio
        self.emitted = (first_frames - vocoder.hift_hold_frames) * vocoder.hift_samples_per_mel_frame
        self.history = torch.zeros(1, vocoder.flow.output_size, self.history_frames, device="cuda")

    def __call__(self) -> None:
        for pair, saved in zip(self.caches, self.saved):
            for row, (committed, context) in zip(pair, saved):
                row.committed_frames = committed
                row.conv_context = context.clone()
        self.vocoder.hop_batch_prefix(self.items, self.caches)
        self.vocoder.hift_step(
            [
                HiftStepRow(history=self.history, emitted_samples=self.emitted, is_final=False)
                for _ in range(HOP_ROWS)
            ]
        )


class FirstHop:
    def __init__(self, vocoder) -> None:
        self.vocoder = vocoder
        self.item = flow_input(vocoder, FIRST_HOP_TOKENS)
        self.frames = causal_hop_frames(self.item, token_mel_ratio=vocoder.flow.token_mel_ratio)

    def __call__(self) -> None:
        cache = self.vocoder.prefix_cache_rows(self.frames)
        try:
            mel = self.vocoder.hop_batch_prefix([self.item], [cache])[0]
            self.vocoder.hift_step([HiftStepRow(history=mel, emitted_samples=0, is_final=False)])
        finally:
            self.vocoder.release_prefix_cache(cache)


def timed(stream_context, step) -> tuple[float, float]:
    with torch.inference_mode(), stream_context:
        start = time.perf_counter()
        step()
        return start, time.perf_counter()


def summary(values: list[float]) -> str:
    ordered = sorted(values)
    return (
        f"p50 {1e3 * ordered[len(ordered) // 2]:7.1f} p90 {1e3 * ordered[int(0.9 * (len(ordered) - 1))]:7.1f}"
        f" mean {1e3 * statistics.mean(values):7.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--prefix-gb", type=float, default=20.0)
    parser.add_argument("--first-prefix-gb", type=float, default=2.0)
    args = parser.parse_args()
    follow_vocoder = build(args.prefix_gb)
    first_vocoder = build(args.first_prefix_gb)
    follow_stream_context = follow_vocoder.stream_context
    with follow_stream_context:
        follow_stream = torch.cuda.current_stream()
    least, greatest = torch.cuda.Stream.priority_range()
    priority = vocoder_decode_stream_priority(torch.cuda)
    contexts = {
        "same stream": torch.cuda.stream(follow_stream),
        "second stream, same priority": torch.cuda.stream(torch.cuda.Stream(priority=0)),
        "second stream, one priority higher": torch.cuda.stream(torch.cuda.Stream(priority=priority)),
    }
    print(f"priority range least {least} greatest {greatest}; first hop stream priority {priority}", flush=True)
    first_hop = FirstHop(first_vocoder)
    for name, step in (("final 4 rows", FinalStep(follow_vocoder)), ("hop 16 rows", HopStep(follow_vocoder))):
        for _ in range(3):
            timed(follow_stream_context, step)
            timed(contexts["second stream, one priority higher"], first_hop)
        alone_follow = [(lambda s, e: e - s)(*timed(follow_stream_context, step)) for _ in range(args.rounds)]
        alone_first = [
            (lambda s, e: e - s)(*timed(contexts["second stream, one priority higher"], first_hop))
            for _ in range(args.rounds)
        ]
        print(f"\n{name}: follow-up step alone {summary(alone_follow)}; first hop alone {summary(alone_first)}")
        for label, context in contexts.items():
            latencies, follows, spans = [], [], []
            for _ in range(args.rounds):
                torch.cuda.synchronize()
                result: dict[str, tuple[float, float]] = {}

                def run_follow() -> None:
                    result["follow"] = timed(follow_stream_context, step)

                def run_first() -> None:
                    time.sleep(LAUNCH_DELAY_S)
                    result["first"] = timed(context, first_hop)

                threads = [threading.Thread(target=run_follow), threading.Thread(target=run_first)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                follow_start, follow_end = result["follow"]
                first_start, first_end = result["first"]
                latencies.append(first_end - first_start)
                follows.append(follow_end - follow_start)
                spans.append(max(follow_end, first_end) - follow_start)
            print(
                f"  first hop 5 ms into it, {label}:\n"
                f"    first hop latency {summary(latencies)}\n"
                f"    follow-up step wall {summary(follows)}\n"
                f"    both {summary(spans)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
