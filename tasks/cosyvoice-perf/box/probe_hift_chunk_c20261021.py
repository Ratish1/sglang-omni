"""HiFT's decode of one mel chunk on the vocoder as serving builds it, from the arm's tree (its omni package): the
torch profiler's device time per chunk, the convolution share and the kernel census (launches per chunk), and seeded
decode outputs saved for a bitwise comparison across trees. 10-21 copy of the 0.5.22 program's probe_hift_convs.py,
with --rows, --channel-alignment (sets the channels-last decode's spectrum alignment before the build, the probe's
own patch) and --save; --compare reads two saved files.

usage: python3 probe_hift_chunk_c20261021.py --checkpoint DIR [--frames 100] [--rows 1] [--runs 20]
           [--channel-alignment N] [--save OUT.pt]
       python3 probe_hift_chunk_c20261021.py --compare A.pt B.pt
"""

import argparse
import importlib
import json
from collections import Counter

import numpy as np
import torch

SAVED_CASES = ((1, 100, True), (4, 100, True), (4, 256, True), (1, 100, False), (4, 256, False))


def source_for(hift, mel):
    f0 = hift.f0_predictor(mel.to(dtype=torch.float64), finalize=True).to(mel)
    source = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
    source, _, _ = hift.m_source(source)
    return source.transpose(1, 2)


def compare(first_path, second_path):
    first, second = torch.load(first_path), torch.load(second_path)
    print(f"A {first['identity']}\nB {second['identity']}")
    for case in first["outputs"]:
        a, b = first["outputs"][case], second["outputs"][case]
        same_shape = a.shape == b.shape
        equal = same_shape and torch.equal(a, b)
        deviation = (a - b).abs().max().item() if same_shape else float("nan")
        print(f"case rows,frames,finalize={case}: shape {tuple(a.shape)} {'equal' if equal else 'DIFFERENT'} max abs {deviation:.3e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--channel-alignment", type=int)
    parser.add_argument("--save")
    parser.add_argument("--compare", nargs=2)
    arguments = parser.parse_args()
    if arguments.compare:
        compare(*arguments.compare)
        return
    stages = importlib.import_module("sglang_omni.models.fun_cosyvoice3.stages")
    channels_last = importlib.import_module("sglang_omni.models.fun_cosyvoice3.hift_channels_last")
    if arguments.channel_alignment is not None:
        channels_last.CHANNEL_ALIGNMENT = arguments.channel_alignment
    flow, hift = stages.load_cosyvoice3_flow_hift(arguments.checkpoint, "cuda", autocast_dtype=None)
    stages.CosyVoice3Vocoder(flow, hift, autocast_dtype=None)
    identity = {"decode": type(hift.decode).__name__, "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
                "channel_alignment": getattr(channels_last, "CHANNEL_ALIGNMENT", None), "module": stages.__file__}
    print(json.dumps(identity))
    samples_per_frame = int(np.prod(hift.upsample_rates)) * hift.istft_params["hop_len"]
    with torch.inference_mode():
        if arguments.save:
            outputs = {}
            for rows, frames, finalize in SAVED_CASES:
                torch.manual_seed(rows * 1000 + frames)
                mel = torch.randn(rows, 80, frames, device="cuda") * 3
                source = torch.randn(rows, 1, frames * samples_per_frame, device="cuda") * 0.1
                outputs[(rows, frames, finalize)] = hift.decode(x=mel, s=source, finalize=finalize).cpu()
            torch.save({"identity": identity, "outputs": outputs}, arguments.save)
            print(f"saved {len(outputs)} cases to {arguments.save}")
        else:
            pass
        torch.manual_seed(0)
        mel = torch.randn(arguments.rows, 80, arguments.frames, device="cuda")
        source = source_for(hift, mel)
        for _ in range(3):
            hift.decode(x=mel, s=source, finalize=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(arguments.runs):
                hift.decode(x=mel, s=source, finalize=True)
            torch.cuda.synchronize()
    time_by_name, count_by_name = Counter(), Counter()
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            time_by_name[event.name] += event.device_time
            count_by_name[event.name] += 1
        else:
            pass
    runs = arguments.runs
    total = sum(time_by_name.values()) / runs / 1000
    conv = [name for name in time_by_name if any(key in name for key in ("conv", "xmma", "cudnn", "gemm"))]
    conv_total = sum(time_by_name[name] for name in conv) / runs / 1000
    launches = sum(count_by_name.values()) / runs
    print(json.dumps({"rows": arguments.rows, "frames": arguments.frames, "gpu_ms_per_chunk": round(total, 3),
                      "conv_ms_per_chunk": round(conv_total, 3), "launches_per_chunk": launches}))
    for name in sorted(time_by_name, key=lambda n: -time_by_name[n])[:24]:
        print(f"  {time_by_name[name] / runs / 1000:7.3f} ms/chunk  {time_by_name[name] / count_by_name[name]:7.1f} us x "
              f"{count_by_name[name] / runs:5.1f}/chunk  {name[:110]}")


if __name__ == "__main__":
    main()
