"""One persistent predictor layer launch inside a cudaProfilerStart/Stop window, for Nsight Compute.

The layer is built with -lineinfo so source counters map to predictor_layer.cuh lines. Run with
PYTHONPATH on the tree that holds predictor_persistent.py.

Usage: ncu --profile-from-start off ... python3 persistent_ncu.py --rows N
"""

import argparse

import torch
from predictor_bench import CACHED, EPS, HEAD_DIM, HIDDEN, KV_HEADS, SLOTS, load_predictor, rope_cache
from sglang.kernels.jit.utils import load_jit, make_cpp_args

from sglang_omni.models.qwen3_tts import predictor_persistent as persistent

WARMUP_LAUNCHES = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = load_predictor(device)
    first = model.layers[0]
    weights = persistent.pack_predictor_layer(
        qkv_weight=first.qkv, output_projection_weight=first.o, gate_up_weight=first.gate_up,
        down_weight=first.down, input_norm_weight=first.input_norm, post_norm_weight=first.post_norm,
        query_norm_weight=first.q_norm, key_norm_weight=first.k_norm,
    )
    layer = persistent.PersistentPredictorLayer(device, rope_cache(device), EPS)
    template_arguments = make_cpp_args(True)
    layer.module = load_jit(
        "qwen3_tts_predictor_layer_lineinfo",
        *template_arguments,
        cuda_files=[str(persistent.PREDICTOR_LAYER_SOURCE)],
        cuda_wrappers=[("predictor_layer", f"qwen3_tts_predictor::PredictorLayer<{template_arguments}>::run")],
        extra_cuda_cflags=["-lineinfo"],
    )
    key_cache = torch.randn(64, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    x = torch.randn(args.rows, HIDDEN, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    for _ in range(WARMUP_LAUNCHES):
        layer.forward(weights, x, out, CACHED, key_cache, value_cache)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    layer.forward(weights, x, out, CACHED, key_cache, value_cache)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("persistent ncu launch done", flush=True)


if __name__ == "__main__":
    main()
