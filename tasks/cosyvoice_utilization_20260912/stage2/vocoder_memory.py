#!/usr/bin/env python3
"""Account for every byte the Fun-CosyVoice3 vocoder stage takes on a card.

Loads the stage the way the factory does, one step at a time, and reports what
each step cost in torch memory and in device memory, so the difference between
the two names what is allocated outside torch (ONNX Runtime arenas, the CUDA
context, cuBLAS and cuDNN workspaces). Then captures Flow CUDA graphs at a
chosen number of shapes, which is the question a shared graph pool answers: if
memory tracks the largest shape the pool is shared, if it tracks the count it is
not.

No server, no benchmark. One process, one card.
"""
from __future__ import annotations

import argparse
import subprocess
from collections import Counter

import torch


def device_free_mb(index: int) -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


class Ledger:
    """Device and torch memory at each step, and what the step cost."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.rows: list[tuple[str, float, float, float, float]] = []
        self.last_device = device_free_mb(index)
        self.last_torch = torch.cuda.memory_reserved() / 2**20

    def mark(self, label: str) -> None:
        torch.cuda.synchronize()
        device = device_free_mb(self.index)
        reserved = torch.cuda.memory_reserved() / 2**20
        self.rows.append(
            (
                label,
                device - self.last_device,
                reserved - self.last_torch,
                device,
                reserved,
            )
        )
        self.last_device, self.last_torch = device, reserved

    def report(self) -> None:
        print(f"\n{'step':34s} {'device MiB':>11s} {'torch MiB':>10s} "
              f"{'outside torch':>14s} {'device total':>13s}")
        for label, device, reserved, total, _ in self.rows:
            print(
                f"{label:34s} {device:11.0f} {reserved:10.0f} "
                f"{device - reserved:14.0f} {total:13.0f}"
            )


def parameter_bytes(module: torch.nn.Module) -> tuple[int, Counter]:
    total = 0
    dtypes: Counter = Counter()
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += tensor.numel() * tensor.element_size()
        dtypes[str(tensor.dtype)] += tensor.numel()
    return total, dtypes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, default=0, help="nvidia-smi card index")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--shapes",
        type=int,
        default=0,
        help="how many of the shipped capture shapes to capture, 0 for none",
    )
    args = parser.parse_args()

    from sglang_omni.models.fun_cosyvoice3 import stages
    from sglang_omni.models.fun_cosyvoice3.config import (
        FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES as SHAPES,
    )

    torch.cuda.init()
    ledger = Ledger(args.index)
    ledger.mark("cuda context")

    flow, hift = stages.load_cosyvoice3_flow_hift(
        args.model, device="cuda:0", fp16=(args.dtype == "float16")
    )
    ledger.mark("flow and hift weights")

    for name, module in (("flow", flow.flow), ("hift", hift)):
        size, dtypes = parameter_bytes(module)
        spread = ", ".join(
            f"{dtype} {count / 1e6:.1f}M" for dtype, count in dtypes.most_common()
        )
        print(f"{name}: {size / 2**20:.0f} MiB resident, {spread}")

    estimator = flow.decoder.estimator
    size, _ = parameter_bytes(estimator)
    print(f"flow.decoder.estimator (the DiT): {size / 2**20:.0f} MiB")

    from sglang_omni.models.fun_cosyvoice3.utils import SpeakerEncoder, SpeechTokenizerV3

    root = stages.resolve_checkpoint(args.model)
    tokenizer = SpeechTokenizerV3(
        f"{root}/speech_tokenizer_v3.onnx", device="cuda:0", intra_op_threads=16
    )
    ledger.mark("onnx speech tokenizer")
    encoder = SpeakerEncoder(f"{root}/campplus.onnx", device="cuda:0", intra_op_threads=16)
    ledger.mark("onnx speaker encoder")
    del tokenizer, encoder

    if args.shapes:
        shapes = SHAPES[: args.shapes]
        runner = stages.FlowCudaGraphRunner(
            flow,
            device=torch.device("cuda:0"),
            autocast_dtype=stages.AUTOCAST_DTYPES[args.dtype],
        )
        runner.capture(shapes)
        flow.attach_cuda_graph_runner(runner)
        widest = max(batch * frames for batch, frames in shapes)
        ledger.mark(f"flow graphs, {len(shapes)} shapes")
        print(
            f"\ncaptured {len(shapes)} shapes, widest batch x frames = {widest}, "
            f"first {shapes[0]}, last {shapes[-1]}"
        )

    ledger.report()


if __name__ == "__main__":
    main()
