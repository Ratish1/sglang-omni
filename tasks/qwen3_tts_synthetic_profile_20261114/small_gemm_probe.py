"""SGLang's small-batch bf16 GEMM (TensorRT-LLM's dsv3 fused A GEMM, swap-AB mma.sync) on the predictor's shapes.

Times each projection alone and the four layer projections as one dependent chain, against cuBLAS,
in the predictor bench's CUDA graph harness. The kernel takes 1 to 16 rows.

Usage: python3 small_gemm_probe.py --out <json>
"""

import argparse
import json

import torch
from predictor_bench import HBM_BYTES_PER_S, load_predictor, time_call

from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import dsv3_fused_a_gemm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    model = load_predictor(device)
    layer = model.layers[0]
    weights = {"qkv": layer.qkv, "o_proj": layer.o, "gate_up": layer.gate_up, "down": layer.down, "lm_head": model.lm_head}
    results = {}
    for rows in (1, 8, 16):
        cell = {}
        inputs = {name: torch.randn(rows, weight.shape[1], device=device, dtype=torch.bfloat16) for name, weight in weights.items()}
        outputs = {name: torch.empty(rows, weight.shape[0], device=device, dtype=torch.bfloat16) for name, weight in weights.items()}
        for name, weight in weights.items():
            floor = weight.numel() * 2 / HBM_BYTES_PER_S * 1e6
            reference = inputs[name].float() @ weight.float().t()
            small = dsv3_fused_a_gemm(inputs[name], weight.t())
            cell[f"{name} max abs vs fp32"] = (small.float() - reference).abs().max().item()
            cell[f"{name} small-batch gemm"] = time_call(lambda: torch.ops.sglang.jit_dsv3_fused_a_gemm(inputs[name], weight.t()))
            cell[f"{name} cublas"] = time_call(lambda: torch.mm(inputs[name], weight.t(), out=outputs[name]))
            cell[f"{name} floor"] = floor
        hidden = torch.randn(rows, 1024, device=device, dtype=torch.bfloat16)

        def small_chain():
            qkv = torch.ops.sglang.jit_dsv3_fused_a_gemm(hidden, layer.qkv.t())
            out = torch.ops.sglang.jit_dsv3_fused_a_gemm(qkv[:, :2048].contiguous(), layer.o.t())
            gate_up = torch.ops.sglang.jit_dsv3_fused_a_gemm(out, layer.gate_up.t())
            return torch.ops.sglang.jit_dsv3_fused_a_gemm(gate_up[:, :3072].contiguous(), layer.down.t())

        def cublas_chain():
            qkv = torch.mm(hidden, layer.qkv.t())
            out = torch.mm(qkv[:, :2048], layer.o.t())
            gate_up = torch.mm(out, layer.gate_up.t())
            return torch.mm(gate_up[:, :3072], layer.down.t())

        cell["layer projections chain, small-batch gemm"] = time_call(small_chain)
        cell["layer projections chain, cublas"] = time_call(cublas_chain)
        cell["layer projections floor"] = sum(w.numel() for w in (layer.qkv, layer.o, layer.gate_up, layer.down)) * 2 / HBM_BYTES_PER_S * 1e6
        results[rows] = cell
        print(json.dumps({rows: cell}, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
