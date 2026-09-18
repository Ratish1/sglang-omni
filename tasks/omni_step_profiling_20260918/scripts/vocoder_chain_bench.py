"""V-e6: the revised vocoder chain, eager and compiled, over every captured key.

The resident chain of vocoder_resident_bench.py with two changes from READOUT_05:
1x1 convs (every residual conv2, the quantizer output_proj) run as matrix multiplies on
(B, T, C), which is the same arithmetic and keeps the layout; and a conv with no history
skips the concatenation with its empty history. The depthwise conv runs resident or
in NCL (--depthwise).

  sweep     every captured key: current and chain, each eager and dynamic-compiled
            (one torch.compile(dynamic=True) per arm, as V2-e1), ms per replay, kernels,
            cuDNN transposes
  split     kernel time by category at six keys for the same four arms
  numerics  V1-e2's streaming decode of real codes against the fp32 decoder, all four
            arms plus today's production setting (only width 8 compiled)

usage: python vocoder_chain_bench.py <sweep|split|numerics> --model DIR [--depthwise ncl]
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn.functional as F
from vocoder_attribution_bench import SPLIT_KEYS, split_replay
from vocoder_resident_bench import (
    BATCHES,
    WIDTHS,
    ResidentDecoder,
    graph_of,
    kernel_names,
    load,
    random_codes,
    replay_ms,
    snr_db,
    stream_decode,
    transposes,
)

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder


class ChainDecoder(ResidentDecoder):
    """Resident chain with pointwise convs as matrix multiplies."""

    def conv(self, conv, x):
        if conv.kernel_size[0] == 1 and conv.groups == 1:
            return F.linear(x, conv.weight[:, :, 0], conv.bias)
        return super().conv(conv, x)

    def causal_conv(self, module, x, state, key):
        if int(module.padding) == 0:
            return self.conv(module.conv, x)
        return super().causal_conv(module, x, state, key)


def build_arms(incremental, device, depthwise_resident: bool) -> dict:
    chain = ChainDecoder(incremental, depthwise_resident=depthwise_resident)
    bf16 = torch.bfloat16
    current_state = lambda b: incremental.init_state(b, device=device, dtype=bf16)
    chain_state = lambda b: chain.init_state(b, device, bf16)
    return {
        "current": (incremental._decode_tensors, current_state),
        "current_dynamic": (
            torch.compile(incremental._decode_tensors, dynamic=True, fullgraph=True),
            current_state,
        ),
        "chain": (chain.decode_tensors, chain_state),
        "chain_dynamic": (
            torch.compile(chain.decode_tensors, dynamic=True, fullgraph=True),
            chain_state,
        ),
    }


def mode_sweep(args, tokenizer, incremental, device) -> None:
    arms = build_arms(incremental, device, args.depthwise == "resident")
    print(
        f"{'width':>5} {'batch':>5} {'arm':>16} {'ms':>8} {'kernels':>8} {'transposes':>10}"
    )
    for width in WIDTHS:
        for batch in BATCHES:
            codes = random_codes(batch, width, device)
            for label, (fn, make_state) in arms.items():
                graph = graph_of(fn, codes, make_state(batch))
                ms = replay_ms(graph, args.reps)
                names = kernel_names(graph.replay)
                print(
                    f"{width:>5} {batch:>5} {label:>16} {ms:>8.3f} {len(names):>8} {transposes(names):>10}",
                    flush=True,
                )
                del graph
            torch.cuda.empty_cache()


def mode_split(args, tokenizer, incremental, device) -> None:
    arms = build_arms(incremental, device, args.depthwise == "resident")
    for width, batch in SPLIT_KEYS:
        codes = random_codes(batch, width, device)
        print(f"\n== width {width} batch {batch}", flush=True)
        for label, (fn, make_state) in arms.items():
            graph = graph_of(fn, codes, make_state(batch))
            ms = replay_ms(graph, args.reps)
            groups, per_name = split_replay(graph)
            total = sum(sum(v) for v in groups.values())
            lines = [f"-- {label}: replay {ms:.3f} ms, kernel sum {total / 1e3:.3f} ms"]
            for group, times in sorted(groups.items(), key=lambda item: -sum(item[1])):
                lines.append(
                    f"   {group:>20} {len(times):>5} kernels {sum(times):>10.1f} us"
                )
            for name, us in sorted(per_name.items(), key=lambda item: -item[1])[
                : args.top
            ]:
                lines.append(f"      {us:>9.1f} us  {name[:120]}")
            print("\n".join(lines), flush=True)
            del graph
        torch.cuda.empty_cache()


def mode_numerics(args, tokenizer, incremental, device) -> None:
    samples = load_seedtts_samples(args.meta, split=args.lang)
    distinct = list({s.ref_audio: s for s in samples}.values())[: args.utterances]
    encoded = tokenizer.encode([s.ref_audio for s in distinct])
    utterances = [
        c.to(device).transpose(0, 1).unsqueeze(0) for c in encoded.audio_codes
    ]
    print("frames per utterance:", [int(u.shape[-1]) for u in utterances])
    truth_decoder = Qwen3TTSIncrementalDecoder(
        copy.deepcopy(incremental._decoder).float()
    )
    arms = build_arms(incremental, device, args.depthwise == "resident")
    production = torch.compile(
        incremental._decode_tensors, dynamic=False, fullgraph=True
    )
    runs = {label: (fn, make_state, None) for label, (fn, make_state) in arms.items()}
    runs["production_w8_compiled"] = (
        incremental._decode_tensors,
        arms["current"][1],
        production,
    )
    shortest = min(int(u.shape[-1]) for u in utterances[:4])
    cohorts = [(f"bs1 utt{i}", u) for i, u in enumerate(utterances)]
    cohorts.append(
        ("bs4 cohort", torch.cat([u[..., :shortest] for u in utterances[:4]], dim=0))
    )
    print(f"{'cohort':>12} {'arm':>24} {'snr_db_min':>10} {'max_abs':>10}")
    for label, codes in cohorts:
        truth = stream_decode(
            incremental,
            truth_decoder._decode_tensors,
            lambda b: truth_decoder.init_state(b, device=device, dtype=torch.float32),
            codes,
        )
        for arm, (step, make_state, compiled_step) in runs.items():
            wave = stream_decode(incremental, step, make_state, codes, compiled_step)
            snrs = [snr_db(wave[row], truth[row]) for row in range(wave.shape[0])]
            print(
                f"{label:>12} {arm:>24} {min(snrs):>10.2f} {(wave - truth).abs().max().item():>10.3e}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sweep", "split", "numerics"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--depthwise", choices=("resident", "ncl"), default="resident")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    torch._dynamo.config.recompile_limit = 256
    tokenizer, incremental = load(args.model, device)
    print(
        f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}, depthwise {args.depthwise}"
    )
    {"sweep": mode_sweep, "split": mode_split, "numerics": mode_numerics}[args.mode](
        args, tokenizer, incremental, device
    )


if __name__ == "__main__":
    main()
