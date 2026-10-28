"""Two #3284 checks on the tree in PYTHONPATH (ours):

capture: in a fresh process, the qkv and gate_up launches at 64 rows run first inside a CUDA graph capture (global
         mode, no eager warmup) with NUM_STAGES forced past the GPU's shared memory, so the refused load and the
         two-stage load both happen during capture; the replay must equal the eager three-stage outputs bit for bit.
mma:     per launch and row block, which MMA instructions the compiled PTX for this GPU uses (mma.sync, the SM 8.x and
         12.x family, or wgmma, Hopper's), at 2 and 3 stages.

usage: PYTHONPATH=TREE python3 predictor_capture_check.py capture | mma
"""

import argparse
import re

import torch
from triton.runtime import jit

from sglang_omni.utils import predictor_layers

from predictor_fit import NAMES, Launches


def capture() -> None:
    device = torch.device("cuda")
    eager = Launches(64, 1, device)
    expected = {}
    for name in ("attention_inputs", "mlp_up"):
        expected[name] = [output.clone() for output in eager.run(name, {})]
    torch.cuda.synchronize()
    captured = Launches(64, 1, device)
    predictor_layers.NUM_STAGES = 6
    fitted_stages = {}
    graph = torch.cuda.CUDAGraph()
    outputs = {}
    with torch.cuda.graph(graph):
        for name in ("attention_inputs", "mlp_up"):
            outputs[name] = captured.run(name, fitted_stages)
    for output_list in outputs.values():
        for output in output_list:
            output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for name in ("attention_inputs", "mlp_up"):
        identical = all(torch.equal(a, b) for a, b in zip(outputs[name], expected[name]))
        print(f"{name} 64 rows, first load inside capture: stages {fitted_stages.get((name, 64))}, replay identical to eager 3 stages {identical}", flush=True)


def mma() -> None:
    device = torch.device("cuda")
    records = []
    original_run = jit.JITFunction.run

    def compile_only(self, *run_args, grid, warmup, **kwargs):
        kernel = original_run(self, *run_args, grid=grid, warmup=True, **kwargs)
        ptx = kernel.asm["ptx"]
        records.append(
            (self.fn.__name__, kwargs.get("BLOCK_M"), kwargs.get("num_stages"),
             len(re.findall(r"\bmma\.sync", ptx)), len(re.findall(r"\bwgmma\.mma_async", ptx)))
        )
        return kernel

    jit.JITFunction.run = compile_only
    for num_stages in (2, 3):
        predictor_layers.NUM_STAGES = num_stages
        for rows in (16, 32, 64):
            launches = Launches(rows, 1, device)
            for name in NAMES:
                launches.run(name, {})
    for kernel_name, block_m, num_stages, mma_sync, wgmma in sorted(set(records), key=str):
        print(f"{kernel_name} BLOCK_M {block_m} stages {num_stages}: mma.sync {mma_sync}, wgmma {wgmma}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "mma"))
    options = parser.parse_args()
    print(f"{torch.cuda.get_device_name()}", flush=True)
    if options.mode == "capture":
        capture()
    else:
        mma()


if __name__ == "__main__":
    main()
