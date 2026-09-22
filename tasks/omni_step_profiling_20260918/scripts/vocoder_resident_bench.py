"""Slice V1 experiments on the real Qwen3-TTS decoder (slices/V1_VOCODER_CHANNELS_LAST.md).

A bench-local resident decoder carries (B, T, C) activations and channels-fastest conv
weights; everything else (transformer, SnakeBeta, ConvNeXt norms and linears) is the
shipped code or the tokenizer's own modules, called in the order of
Qwen3TTSIncrementalDecoder._decode_tensors.

  dispatch    V1-e0: kernels and transposes of one decode, state and weight-copy sizes
  timing      V1-e1: every captured key, CUDA graph replay ms, kernel and transpose counts
  numerics    V1-e2: real codes, streaming chunk schedule, SNR against an fp32 decoder
  fulldecode  V1-e4: non-streaming tokenizer.decode with NCL vs in-place channels-last weights

usage: python vocoder_resident_bench.py <mode> --model DIR
"""

from __future__ import annotations

import argparse
import copy
import statistics

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalDecoder,
    _incremental_transformer,
)

WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
BATCHES = (1, 2, 4, 8)
COMPILED_WIDTH = 8
CHUNK_SCHEDULE = (1, 2, 4)
STEADY_STRIDE = 8
TRANSPOSE_KERNELS = ("nchwToNhwc", "nhwcToNchw")


def channels_fastest(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(1, 2).contiguous().transpose(1, 2)


class ResidentDecoder:
    """The incremental decode with (B, T, C) activations and state."""

    def __init__(
        self, incremental: Qwen3TTSIncrementalDecoder, *, depthwise_resident: bool
    ) -> None:
        self.incremental = incremental
        self.decoder = incremental._decoder
        # the copies hang off the conv modules under a per-arm name, so
        # torch.compile reads them as module attributes; two arms can coexist.
        self.attr = (
            "_bench_weight_resident" if depthwise_resident else "_bench_weight_dw_ncl"
        )
        self.weight_bytes = 0
        self.num_weights = 0
        for module in self.decoder.modules():
            if not isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                continue
            weight = None
            if module.groups == 1 or depthwise_resident:
                weight = channels_fastest(module.weight.detach())
                self.weight_bytes += weight.numel() * weight.element_size()
                self.num_weights += 1
            setattr(module, self.attr, weight)
        self.count_strided = False
        self.strided_outputs = 0

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        state = self.incremental.init_state(batch, device=device, dtype=dtype)
        return to_time_major(state)

    def conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        weight = getattr(conv, self.attr)
        if weight is None:
            y = F.conv1d(
                x.transpose(1, 2).contiguous(),
                conv.weight,
                conv.bias,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )
            return y.transpose(1, 2).contiguous()
        y = F.conv1d(
            x.transpose(1, 2),
            weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        ).transpose(1, 2)
        if self.count_strided:
            self.strided_outputs += not y.is_contiguous()
        return y

    def causal_conv(self, module, x, state, key):
        history_size = int(module.padding)
        combined = torch.cat((state.conv_histories[key], x), dim=1)
        output = self.conv(module.conv, combined)
        state.conv_histories[key] = combined[
            :, combined.shape[1] - history_size :
        ].clone()
        return output

    def transconv(self, module, x, state, key):
        conv = module.conv
        output = F.conv_transpose1d(
            x.transpose(1, 2),
            getattr(conv, self.attr),
            None,
            conv.stride,
            conv.padding,
            conv.output_padding,
            conv.groups,
            conv.dilation,
        ).transpose(1, 2)
        if self.count_strided:
            self.strided_outputs += not output.is_contiguous()
        overlap = state.transconv_overlaps[key]
        output[:, : overlap.shape[1]] += overlap
        emit = int(x.shape[1]) * int(conv.stride[0])
        state.transconv_overlaps[key] = output[:, emit:].clone()
        emitted = output[:, :emit]
        if conv.bias is not None:
            emitted = emitted + conv.bias
        return emitted

    @staticmethod
    def snake(module, x):
        return module(x.transpose(1, 2)).transpose(1, 2)

    def quantizer(self, codes):
        quantizer = self.decoder.quantizer
        semantic = quantizer.n_q_semantic

        def residual(rvq, layer_codes):
            summed = rvq.vq.decode(layer_codes.transpose(0, 1))
            return self.conv(rvq.output_proj, summed.transpose(1, 2))

        hidden = residual(quantizer.rvq_first, codes[:, :semantic])
        if codes.shape[1] > semantic:
            hidden = hidden + residual(quantizer.rvq_rest, codes[:, semantic:])
        return hidden

    def decode_tensors(self, codes, state):
        hidden = self.quantizer(codes)
        hidden = self.causal_conv(self.decoder.pre_conv, hidden, state, "pre_conv")
        hidden = _incremental_transformer(self.decoder.pre_transformer, hidden, state)
        for stage_index, blocks in enumerate(self.decoder.upsample):
            hidden = self.transconv(
                blocks[0], hidden, state, f"upsample.{stage_index}.transconv"
            )
            convnext = blocks[1]
            key = f"upsample.{stage_index}.convnext"
            residual = hidden
            hidden = self.causal_conv(convnext.dwconv, hidden, state, f"{key}.dwconv")
            hidden = convnext.pwconv2(
                convnext.act(convnext.pwconv1(convnext.norm(hidden)))
            )
            hidden = residual + convnext.gamma * hidden
        wave = self.causal_conv(self.decoder.decoder[0], hidden, state, "decoder.0")
        for block_index, block in enumerate(self.decoder.decoder[1:-2], start=1):
            wave = self.snake(block.block[0], wave)
            wave = self.transconv(
                block.block[1], wave, state, f"decoder.{block_index}.transconv"
            )
            for unit_index, unit in enumerate(block.block[2:]):
                key = f"decoder.{block_index}.residual.{unit_index}"
                inner = self.snake(unit.act1, wave)
                inner = self.causal_conv(unit.conv1, inner, state, f"{key}.conv1")
                inner = self.snake(unit.act2, inner)
                inner = self.causal_conv(unit.conv2, inner, state, f"{key}.conv2")
                wave = inner + wave
        wave = self.snake(self.decoder.decoder[-2], wave)
        wave = self.causal_conv(self.decoder.decoder[-1], wave, state, "decoder.final")
        return wave.transpose(1, 2).clamp(min=-1, max=1)


def to_time_major(state):
    state.conv_histories = {
        k: v.transpose(1, 2).contiguous() for k, v in state.conv_histories.items()
    }
    state.transconv_overlaps = {
        k: v.transpose(1, 2).contiguous() for k, v in state.transconv_overlaps.items()
    }
    return state


def advance(incremental, state, frames: int) -> None:
    """The bookkeeping of Qwen3TTSIncrementalDecoder.decode after its tensor step."""
    state.transformer_context_length = min(
        incremental.state_spec().retained_context,
        state.transformer_context_length + frames,
    )
    state.advance(frames)


def load(model: str, device: torch.device):
    tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
        model, device=str(device), dtype="bfloat16", attn_implementation=None
    )
    incremental = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder)
    return tokenizer, incremental


def random_codes(batch: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2048, (batch, 16, width), device=device)


def graph_of(fn, codes, state):
    """fn(codes, state) captured once after warmups on a side stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            fn(codes, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        fn(codes, state)
    return graph


def replay_ms(graph, reps: int) -> float:
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def kernel_names(run) -> list[str]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        run()
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def transposes(names: list[str]) -> int:
    return sum(any(t in name for t in TRANSPOSE_KERNELS) for name in names)


def mode_dispatch(args, tokenizer, incremental, device) -> None:
    spec = incremental.state_spec()
    print(
        f"state_spec: {len(spec.conv_histories)} conv histories, {len(spec.transconv_overlaps)} "
        f"transconv overlaps, {spec.num_layers} layers x 2 kv, retained {spec.retained_context}"
    )
    for depthwise_resident in (True, False):
        resident = ResidentDecoder(incremental, depthwise_resident=depthwise_resident)
        print(
            f"\n== depthwise {'resident' if depthwise_resident else 'ncl'}: "
            f"{resident.num_weights} weight copies, {resident.weight_bytes / 2**20:.1f} MiB"
        )
        for batch in (1, 8):
            codes = random_codes(batch, STEADY_STRIDE, device)
            with torch.inference_mode():
                base_state = incremental.init_state(
                    batch, device=device, dtype=torch.bfloat16
                )
                states = [base_state.clone() for _ in range(2)]
                resident_states = [to_time_major(base_state.clone()) for _ in range(2)]
                reference = incremental._decode_tensors(codes, states[0])
                resident.count_strided = True
                resident.strided_outputs = 0
                output = resident.decode_tensors(codes, resident_states[0])
                strided = resident.strided_outputs
                resident.count_strided = False
                current_names = kernel_names(
                    lambda: incremental._decode_tensors(codes, states[1])
                )
                resident_names = kernel_names(
                    lambda: resident.decode_tensors(codes, resident_states[1])
                )
            print(
                f"bs {batch}: kernels current {len(current_names)} (transposes {transposes(current_names)}), "
                f"resident {len(resident_names)} (transposes {transposes(resident_names)}), "
                f"strided conv outputs {strided}, output max abs vs current "
                f"{(output.float() - reference.float()).abs().max().item():.3e}"
            )
            for name in sorted(
                set(n for n in resident_names if any(t in n for t in TRANSPOSE_KERNELS))
            ):
                print(f"    resident transpose kernel: {name[:120]}")


def mode_timing(args, tokenizer, incremental, device) -> None:
    arms = {
        "current": (
            incremental._decode_tensors,
            lambda b: incremental.init_state(b, device=device, dtype=torch.bfloat16),
        ),
    }
    for depthwise_resident, label in ((True, "resident"), (False, "resident_dw_ncl")):
        resident = ResidentDecoder(incremental, depthwise_resident=depthwise_resident)
        arms[label] = (
            resident.decode_tensors,
            lambda b, r=resident: r.init_state(b, device, torch.bfloat16),
        )
    compiled = {
        "current_compiled": (
            torch.compile(incremental._decode_tensors, dynamic=False, fullgraph=True),
            arms["current"][1],
        ),
        "resident_compiled": (
            torch.compile(arms["resident"][0], dynamic=False, fullgraph=True),
            arms["resident"][1],
        ),
    }
    print(
        f"{'width':>5} {'batch':>5} {'arm':>18} {'ms':>8} {'kernels':>8} {'transposes':>10}"
    )
    for width in WIDTHS:
        for batch in BATCHES:
            codes = random_codes(batch, width, device)
            keyed = dict(arms)
            if width == COMPILED_WIDTH:
                keyed.update(compiled)
            for label, (fn, make_state) in keyed.items():
                state = make_state(batch)
                graph = graph_of(fn, codes, state)
                ms = replay_ms(graph, args.reps)
                names = kernel_names(graph.replay)
                print(
                    f"{width:>5} {batch:>5} {label:>18} {ms:>8.3f} {len(names):>8} {transposes(names):>10}",
                    flush=True,
                )
                del graph
            torch.cuda.empty_cache()


def snr_db(candidate: torch.Tensor, truth: torch.Tensor) -> float:
    noise = (candidate.double() - truth.double()).pow(2).sum()
    return float(10 * torch.log10(truth.double().pow(2).sum() / noise.clamp_min(1e-30)))


def chunk_widths(total: int) -> list[int]:
    widths, used = [], 0
    for width in (*CHUNK_SCHEDULE, *([STEADY_STRIDE] * total)):
        if used >= total:
            break
        widths.append(min(width, total - used))
        used += widths[-1]
    return widths


def stream_decode(
    incremental, step, make_state, codes, compiled_step=None
) -> torch.Tensor:
    """Decode codes chunk by chunk with the production schedule; compiled_step takes width-8 chunks."""
    pieces, offset = [], 0
    with torch.inference_mode():
        state = make_state(codes.shape[0])
        for width in chunk_widths(int(codes.shape[-1])):
            chunk = codes[..., offset : offset + width].contiguous()
            fn = (
                compiled_step
                if (compiled_step is not None and width == COMPILED_WIDTH)
                else step
            )
            pieces.append(fn(chunk, state).float())
            advance(incremental, state, width)
            offset += width
    return torch.cat(pieces, dim=-1)


def mode_numerics(args, tokenizer, incremental, device) -> None:
    samples = load_seedtts_samples(args.meta, split=args.lang)[: args.utterances]
    encoded = tokenizer.encode([s.ref_audio for s in samples])
    utterances = [
        c.to(device).transpose(0, 1).unsqueeze(0) for c in encoded.audio_codes
    ]
    print("frames per utterance:", [int(u.shape[-1]) for u in utterances])

    truth_decoder = Qwen3TTSIncrementalDecoder(
        copy.deepcopy(incremental._decoder).float()
    )
    resident = ResidentDecoder(incremental, depthwise_resident=args.depthwise_resident)
    bf16 = torch.bfloat16
    arms = {
        "current": (
            incremental._decode_tensors,
            lambda b: incremental.init_state(b, device=device, dtype=bf16),
            None,
        ),
        "current_compiled": (
            incremental._decode_tensors,
            lambda b: incremental.init_state(b, device=device, dtype=bf16),
            torch.compile(incremental._decode_tensors, dynamic=False, fullgraph=True),
        ),
        "resident": (
            resident.decode_tensors,
            lambda b: resident.init_state(b, device, bf16),
            None,
        ),
        "resident_compiled": (
            resident.decode_tensors,
            lambda b: resident.init_state(b, device, bf16),
            torch.compile(resident.decode_tensors, dynamic=False, fullgraph=True),
        ),
    }
    cohorts = [(f"bs1 utt{i}", u) for i, u in enumerate(utterances)]
    shortest = min(int(u.shape[-1]) for u in utterances[:4])
    cohorts.append(
        ("bs4 cohort", torch.cat([u[..., :shortest] for u in utterances[:4]], dim=0))
    )
    print(f"{'cohort':>12} {'arm':>18} {'snr_db_min':>10} {'max_abs':>10}")
    for label, codes in cohorts:
        truth = stream_decode(
            incremental,
            truth_decoder._decode_tensors,
            lambda b: truth_decoder.init_state(b, device=device, dtype=torch.float32),
            codes,
        )
        for arm, (step, make_state, compiled_step) in arms.items():
            wave = stream_decode(incremental, step, make_state, codes, compiled_step)
            snrs = [snr_db(wave[row], truth[row]) for row in range(wave.shape[0])]
            print(
                f"{label:>12} {arm:>18} {min(snrs):>10.2f} {(wave - truth).abs().max().item():>10.3e}",
                flush=True,
            )


def mode_fulldecode(args, tokenizer, incremental, device) -> None:
    samples = load_seedtts_samples(args.meta, split=args.lang)[: args.utterances]
    encoded = tokenizer.encode([s.ref_audio for s in samples])
    joined = torch.cat(list(encoded.audio_codes), dim=0)
    convs = [
        m
        for m in tokenizer.model.decoder.modules()
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d))
    ]
    original = {m: m.weight.data for m in convs}
    for seconds in (10, 30):
        frames = int(seconds * 12.5)
        codes = joined[:frames]
        assert codes.shape[0] == frames, f"only {joined.shape[0]} frames available"
        for batch in (1, 8):
            items = [{"audio_codes": codes} for _ in range(batch)]
            results = {}
            for layout in ("ncl", "channels_fastest"):
                for m in convs:
                    m.weight.data = (
                        original[m]
                        if layout == "ncl"
                        else channels_fastest(original[m])
                    )
                with torch.inference_mode():
                    for _ in range(2):
                        tokenizer.decode(items)
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(args.reps):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        wavs, _ = tokenizer.decode(items)
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end))
                results[layout] = (statistics.median(times), torch.as_tensor(wavs[0]))
            for m in convs:
                m.weight.data = original[m]
            diff = (
                (results["ncl"][1].float() - results["channels_fastest"][1].float())
                .abs()
                .max()
                .item()
            )
            print(
                f"{seconds:>3} s bs {batch}: ncl {results['ncl'][0]:.2f} ms, channels_fastest "
                f"{results['channels_fastest'][0]:.2f} ms, waveform max abs {diff:.3e}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("dispatch", "timing", "numerics", "fulldecode")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument(
        "--depthwise-resident", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    tokenizer, incremental = load(args.model, device)
    print(
        f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}"
    )
    {
        "dispatch": mode_dispatch,
        "timing": mode_timing,
        "numerics": mode_numerics,
        "fulldecode": mode_fulldecode,
    }[args.mode](args, tokenizer, incremental, device)


if __name__ == "__main__":
    main()
