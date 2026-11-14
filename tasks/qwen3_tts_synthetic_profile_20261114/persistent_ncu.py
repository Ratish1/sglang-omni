"""One persistent predictor launch inside a cudaProfilerStart/Stop window, for Nsight Compute.

The kernel is built with -lineinfo so source counters map to predictor_layer.cuh lines, and runs
the first --layers predictor layers in one launch. Run with PYTHONPATH on the tree that holds
predictor_persistent.py.

Usage: ncu --profile-from-start off ... python3 persistent_ncu.py --rows N --layers L
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
    parser.add_argument("--layers", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = load_predictor(device)
    weights = persistent.stack_predictor_layers(
        [
            persistent.pack_predictor_layer(
                qkv_weight=each.qkv, output_projection_weight=each.o, gate_up_weight=each.gate_up,
                down_weight=each.down, input_norm_weight=each.input_norm, post_norm_weight=each.post_norm,
                query_norm_weight=each.q_norm, key_norm_weight=each.k_norm,
            )
            for each in model.layers[: args.layers]
        ]
    )
    layer = persistent.PersistentPredictorLayers(device, rope_cache(device), EPS)
    is_flagged = args.rows <= persistent.FLAGGED_MAX_ROWS
    template_arguments = make_cpp_args(True, is_flagged)
    layer.modules = {
        is_flagged: load_jit(
            "qwen3_tts_predictor_layer_lineinfo",
            *template_arguments,
            cuda_files=[str(persistent.PREDICTOR_LAYER_SOURCE)],
            cuda_wrappers=[("predictor_layer", f"qwen3_tts_predictor::PredictorLayer<{template_arguments}>::run")],
            extra_cuda_cflags=["-lineinfo"],
        )
    }
    key_cache = torch.randn(args.layers, 64, SLOTS, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
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
