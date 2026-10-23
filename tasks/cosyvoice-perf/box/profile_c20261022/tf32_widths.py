"""TF32 at the tokenizer's widths (10-23): ONNX Runtime's CUDA provider options as main and upstream CosyVoice create
the session; then a 1280 x 1280 projection (the tokenizer's linear shape) in float32 with TF32 matmuls on and off, at
true reference widths and at the prefill ladder widths they pad to, each row against a float64 product: the largest
relative error per width and whether a row's result changes with the width it is padded to.
usage: CUDA_VISIBLE_DEVICES=<card> python3 tf32_widths.py"""

import bisect
import os

import onnxruntime
import torch
from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes

from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def main() -> None:
    model_path = os.path.join(resolve_checkpoint(MODEL), "speech_tokenizer_v3.onnx")
    for label, providers in (
        ("main", [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]),
        ("upstream", ["CUDAExecutionProvider"]),
    ):
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        options = session.get_provider_options()["CUDAExecutionProvider"]
        print(label, {key: options[key] for key in sorted(options) if "tf32" in key or "conv" in key or "math" in key})
        del session
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    weight = torch.randn(1280, 1280, device=device) / 1280**0.5
    widths = [*generate_prefill_cuda_graph_batch_sizes(2999), 3000]
    rows = torch.randn(3000, 1280, device=device)
    print("true width -> graph width | max relative error vs float64: fp32 true, tf32 true, tf32 padded | tf32 rows equal true vs padded")
    for true_width in (37, 101, 157, 253, 377, 511, 763):
        graph_width = widths[bisect.bisect_left(widths, true_width)]
        x = rows[:true_width]
        padded = torch.cat((x, torch.zeros(graph_width - true_width, 1280, device=device)))
        exact = x.double() @ weight.double()
        scale = exact.abs().max()
        results = {}
        for allow_tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            results[(allow_tf32, "true")] = x @ weight
            results[(allow_tf32, "padded")] = (padded @ weight)[:true_width]
        torch.backends.cuda.matmul.allow_tf32 = False
        error = lambda value: float(((value.double() - exact).abs().max() / scale))
        print(
            f"{true_width} -> {graph_width} | {error(results[(False, 'true')]):.2e} {error(results[(True, 'true')]):.2e} "
            f"{error(results[(True, 'padded')]):.2e} | {torch.equal(results[(True, 'true')], results[(True, 'padded')])}"
            f" (fp32 equal {torch.equal(results[(False, 'true')], results[(False, 'padded')])})"
        )


if __name__ == "__main__":
    main()
