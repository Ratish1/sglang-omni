# SPDX-License-Identifier: Apache-2.0
"""#2406 probe on real weights: cached hop against the whole-history hop for
several hop plans, the hop step time of both, and a mixed step's cost.

usage: python3 prefix_cache_probe.py --out <json> [--eager]
Runs from the arm's tree (PYTHONPATH), the vocoder built by its own factory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    create_vocoder_executor,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_MEL_RATIO,
    pad_flow_prompt_to_hop,
)

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def snr_db(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).pow(2).mean().item()
    signal = expected.float().pow(2).mean().item()
    return math.inf if error == 0 else 10 * math.log10(signal / error)


def hop_plan(hop: int, max_hop: int, count: int) -> list[tuple[int, int]]:
    offset, length, plan = 0, hop, []
    for _ in range(count):
        plan.append((offset, length))
        offset += length
        length = min(max_hop, length * 2)
    return plan


def make_streams(vocoder, prompt_lengths, total_tokens, hop, seed):
    flow = vocoder.flow
    generator = torch.Generator().manual_seed(seed)
    vocab = int(flow.input_embedding.num_embeddings)
    streams = []
    for prompt_length in prompt_lengths:
        prompt_token = torch.randint(
            0, vocab, (1, prompt_length), generator=generator, dtype=torch.int32
        )
        prompt_feat = (
            torch.randn(1, prompt_length * 2, flow.output_size, generator=generator) * 2
            - 5
        )
        prompt_token, prompt_feat = pad_flow_prompt_to_hop(
            prompt_token, prompt_feat, hop_len=hop
        )
        streams.append(
            dict(
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                tokens=torch.randint(
                    0, vocab, (1, total_tokens), generator=generator, dtype=torch.int32
                ),
                embedding=torch.randn(
                    1, flow.spk_embed_affine_layer.in_features, generator=generator
                ),
                cache=None,
            )
        )
    return streams


def items_at(streams, offset, length):
    return [
        FlowBatchInput(
            token=stream["tokens"][:, : offset + length + PRE_LOOKAHEAD_LEN],
            prompt_token=stream["prompt_token"],
            prompt_feat=stream["prompt_feat"],
            embedding=stream["embedding"],
        )
        for stream in streams
    ]


def frames_of(item):
    return (
        int(item.prompt_token.shape[1]) + int(item.token.shape[1]) - PRE_LOOKAHEAD_LEN
    ) * TOKEN_MEL_RATIO


def cached_hop(vocoder, streams, items):
    for stream, item in zip(streams, items, strict=True):
        if stream["cache"] is None:
            stream["cache"] = vocoder.prefix_cache_rows(frames_of(item))
            assert stream["cache"] is not None
        else:
            assert vocoder.grow_prefix_cache(stream["cache"], frames_of(item))
    return vocoder.hop_batch_prefix(items, [stream["cache"] for stream in streams])


def release(vocoder, streams):
    for stream in streams:
        vocoder.release_prefix_cache(stream["cache"])
        stream["cache"] = None


def timed(call):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1e3


def exactness(vocoder, hop, max_hop, prompt_lengths, hops):
    plan = hop_plan(hop, max_hop, hops)
    total = sum(length for _, length in plan) + PRE_LOOKAHEAD_LEN
    streams = make_streams(
        vocoder, prompt_lengths, total, hop, seed=hop * 1000 + max_hop
    )
    rows = []
    for offset, length in plan:
        items = items_at(streams, offset, length)
        reference = vocoder.hop_batch(items)
        cached = cached_hop(vocoder, streams, items)
        start, end = offset * TOKEN_MEL_RATIO, (offset + length) * TOKEN_MEL_RATIO
        for index, (expected, actual) in enumerate(zip(reference, cached, strict=True)):
            expected, actual = expected[:, :, start:end], actual[:, :, start:end]
            prefix_frames = int(streams[index]["prompt_token"].shape[1]) * 2 + start
            rows.append(
                dict(
                    hop=hop,
                    max_hop=max_hop,
                    offset=offset,
                    length=length,
                    row=index,
                    prefix_frames=prefix_frames,
                    prefix_on_chunk=prefix_frames % 50 == 0,
                    equal=bool(torch.equal(actual, expected)),
                    max_abs=float((actual - expected).abs().max()),
                    snr_db=snr_db(actual, expected),
                )
            )
    release(vocoder, streams)
    return rows


def step_times(vocoder, prompt_lengths, rounds):
    plan = hop_plan(25, 100, 5)
    total = sum(length for _, length in plan) + PRE_LOOKAHEAD_LEN
    whole: dict[int, list[float]] = {}
    cached: dict[int, list[float]] = {}
    mixed: dict[int, list[float]] = {}
    for round_index in range(rounds):
        streams = make_streams(vocoder, prompt_lengths, total, 25, seed=round_index)
        for hop_index, (offset, length) in enumerate(plan):
            items = items_at(streams, offset, length)
            _, whole_ms = timed(lambda: vocoder.hop_batch(items))
            whole.setdefault(hop_index, []).append(whole_ms)
            if hop_index == 3:
                saved = [
                    (
                        s["cache"][0].frames,
                        s["cache"][1].frames,
                        s["cache"][0].conv_context.clone(),
                        s["cache"][1].conv_context.clone(),
                    )
                    for s in streams
                ]
                for kept in (12, 8, 4):

                    def mixed_step():
                        vocoder.hop_batch_prefix(
                            items[:kept], [s["cache"] for s in streams[:kept]]
                        )
                        vocoder.hop_batch(items[kept:])

                    for s, item in zip(streams[:kept], items[:kept], strict=True):
                        assert vocoder.grow_prefix_cache(s["cache"], frames_of(item))
                    _, mixed_ms = timed(mixed_step)
                    mixed.setdefault(kept, []).append(mixed_ms)
                    for s, (f0, f1, c0, c1) in zip(streams, saved, strict=True):
                        s["cache"][0].frames, s["cache"][1].frames = f0, f1
                        s["cache"][0].conv_context, s["cache"][1].conv_context = c0, c1
            else:
                pass
            _, cached_ms = timed(lambda: cached_hop(vocoder, streams, items))
            cached.setdefault(hop_index, []).append(cached_ms)
        release(vocoder, streams)
    return dict(
        plan=plan,
        whole_ms={k: statistics.median(v) for k, v in whole.items()},
        cached_ms={k: statistics.median(v) for k, v in cached.items()},
        mixed_hop3_ms={k: statistics.median(v) for k, v in mixed.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    scheduler = create_vocoder_executor(
        MODEL,
        device="cuda:0",
        enable_flow_cuda_graph=False,
        enable_dit_torch_compile=not args.eager,
    )
    vocoder = scheduler.vocoder
    pool = vocoder.flow.prefix_pool
    result = dict(
        eager=args.eager,
        pool_frames=pool.free_frames,
        device_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
    )
    prompt_lengths = [37, 61, 80, 113]
    with torch.inference_mode(), vocoder.stream_context:
        result["exactness"] = []
        for hop, max_hop in ((25, 100), (25, 60), (10, 40), (20, 80), (15, 60)):
            result["exactness"] += exactness(vocoder, hop, max_hop, prompt_lengths, 5)
        sixteen = [37, 61, 80, 113, 50, 75, 98, 140, 44, 66, 90, 120, 55, 70, 85, 105]
        result["step"] = step_times(vocoder, sixteen, args.rounds)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1, default=str)
    for row in result["exactness"]:
        print(
            f"hop {row['hop']}/{row['max_hop']} offset {row['offset']:4d} len {row['length']:3d} "
            f"row {row['row']} prefix {row['prefix_frames']:4d} aligned {row['prefix_on_chunk']!s:5} "
            f"equal {row['equal']!s:5} max_abs {row['max_abs']:.4g} snr {row['snr_db']:.1f}"
        )
    print(json.dumps(result["step"], indent=1, default=str))


if __name__ == "__main__":
    main()
