"""Slice V layout experiment on the real decoder's convs at their decode inputs.

Records every conv call (nn.Conv1d module calls and F.conv_transpose1d calls) of one
incremental decode per (batch, fresh_frames), then times each call three ways:
  current   what the decode does today (NCL tensors, cuDNN transposes per call);
  weight    weight laid out channels-last once, input converted per call;
  resident  weight and input both already channels-last (a chain kept in that layout).
Per call: median us over --reps, output bytes against current (torch.equal, max abs).

usage: python vocoder_layout_bench.py --model DIR --shapes 1:1 1:64 8:4 --reps 50
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_tts import incremental_codec
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder

NUM_QUANTIZERS = 16
CODE_RANGE = 1024


def record_calls(decoder, codes, state) -> list[dict]:
    calls: list[dict] = []
    hooks = []
    for module in decoder._decoder.modules():
        if isinstance(module, torch.nn.Conv1d):
            hooks.append(
                module.register_forward_pre_hook(
                    lambda m, inputs: calls.append(
                        {"kind": "conv", "module": m, "input": inputs[0].clone()}
                    )
                )
            )
    original = incremental_codec.F.conv_transpose1d

    def recording_conv_transpose1d(inputs, weight, **kwargs):
        calls.append({"kind": "transpose", "input": inputs.clone(), "weight": weight, "kwargs": kwargs})
        return original(inputs, weight, **kwargs)

    incremental_codec.F.conv_transpose1d = recording_conv_transpose1d
    try:
        with torch.inference_mode():
            decoder.decode(codes, state.clone())
    finally:
        incremental_codec.F.conv_transpose1d = original
        for hook in hooks:
            hook.remove()
    return calls


def variants(call: dict):
    """Three callables over the same call and inputs prepared outside the timing."""
    x = call["input"]
    x4 = x.unsqueeze(2)
    x4_cl = x4.contiguous(memory_format=torch.channels_last)
    if call["kind"] == "conv":
        module = call["module"]
        weight4_cl = module.weight.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        args = dict(
            bias=module.bias,
            stride=(1, module.stride[0]),
            padding=(0, module.padding[0]) if isinstance(module.padding, tuple) else module.padding,
            dilation=(1, module.dilation[0]),
            groups=module.groups,
        )
        current = lambda: module(x)
        weight = lambda: F.conv2d(x4.contiguous(memory_format=torch.channels_last), weight4_cl, **args).squeeze(2)
        resident = lambda: F.conv2d(x4_cl, weight4_cl, **args)
    else:
        kwargs = call["kwargs"]
        weight4_cl = call["weight"].unsqueeze(2).contiguous(memory_format=torch.channels_last)
        args = dict(
            bias=kwargs.get("bias"),
            stride=(1, kwargs["stride"][0]),
            padding=(0, kwargs["padding"][0]),
            output_padding=(0, kwargs["output_padding"][0]),
            groups=kwargs["groups"],
            dilation=(1, kwargs["dilation"][0]),
        )
        current = lambda: F.conv_transpose1d(x, call["weight"], **kwargs)
        weight = lambda: F.conv_transpose2d(
            x4.contiguous(memory_format=torch.channels_last), weight4_cl, **args
        ).squeeze(2)
        resident = lambda: F.conv_transpose2d(x4_cl, weight4_cl, **args)
    return current, weight, resident


def median_us(fn, reps: int, calls_per_replay: int = 10) -> float:
    """Device time per call: calls captured in a CUDA graph, so host launch gaps drop out."""
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls_per_replay):
                fn()
        times = []
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1e3 / calls_per_replay)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--shapes", nargs="+", required=True, help="batch:fresh_frames")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
        args.model, device=str(device), dtype="bfloat16", attn_implementation=None
    )
    decoder = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder)
    torch.manual_seed(0)
    print(f"device {torch.cuda.get_device_name(device)}, torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}")
    for item in args.shapes:
        batch, frames = (int(v) for v in item.split(":"))
        codes = torch.randint(0, CODE_RANGE, (batch, NUM_QUANTIZERS, frames), device=device)
        state = decoder.init_state(batch, device=device, dtype=torch.bfloat16)
        calls = record_calls(decoder, codes, state)
        totals = [0.0, 0.0, 0.0]
        exact = [0, 0]
        print(f"\n== batch {batch} fresh_frames {frames}: {len(calls)} conv calls")
        print(f"{'kind':10s}{'input':>22s}{'weight':>22s}{'groups':>7s}{'current':>10s}{'weight':>10s}{'resident':>10s}  weight==  resident==  max abs")
        for call in calls:
            current, weight, resident = variants(call)
            with torch.inference_mode():
                reference = current()
                out_weight = weight()
                out_resident = resident().squeeze(2)
            times = [median_us(fn, args.reps) for fn in (current, weight, resident)]
            for index, value in enumerate(times):
                totals[index] += value
            same_weight = torch.equal(reference, out_weight)
            same_resident = torch.equal(reference, out_resident)
            exact[0] += same_weight
            exact[1] += same_resident
            max_abs = max(
                (reference.float() - out_weight.float()).abs().max().item(),
                (reference.float() - out_resident.float()).abs().max().item(),
            )
            w = call["module"].weight if call["kind"] == "conv" else call["weight"]
            groups = call["module"].groups if call["kind"] == "conv" else call["kwargs"]["groups"]
            print(
                f"{call['kind']:10s}{str(tuple(call['input'].shape)):>22s}{str(tuple(w.shape)):>22s}{groups:>7d}"
                f"{times[0]:>10.1f}{times[1]:>10.1f}{times[2]:>10.1f}  {str(same_weight):8s}{str(same_resident):12s}{max_abs:.2e}"
            )
        print(
            f"total us: current {totals[0]:.1f}, weight {totals[1]:.1f}, resident {totals[2]:.1f}; "
            f"bit-exact calls: weight {exact[0]}/{len(calls)}, resident {exact[1]}/{len(calls)}"
        )


if __name__ == "__main__":
    main()
