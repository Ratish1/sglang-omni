# SPDX-License-Identifier: Apache-2.0
"""How close a tree's final (leftover) Flow mel in bf16 is to the same tree's float32 final
over the same inputs: DiT compile and graphs off, so only the eager numerics differ. Run it
in two trees with the same seed and compare the relative errors.

    cd <tree> && python final_numerics.py --out <json>
"""

from __future__ import annotations

import argparse
import json

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput

CASES = (("1 row", (100,)), ("4 rows", (60, 100, 140, 180)), ("1 long row", (480,)))
PROMPT_TOKENS = 75


def vocoder(model: str, dtype: str):
    kwargs = dict(
        enable_flow_prefix_cuda_graph=False,
        flow_prefix_cache_gb=0.0,
        enable_dit_torch_compile=False,
        enable_flow_cuda_graph=False,
    )
    if (
        "enable_flow_final_cuda_graph"
        in stages.create_vocoder_executor.__code__.co_varnames
    ):
        kwargs["enable_flow_final_cuda_graph"] = False
    else:
        pass
    return stages.create_vocoder_executor(
        model, device="cuda", gpu_id=0, dtype=dtype, **kwargs
    ).vocoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    half = vocoder(args.model, "bfloat16")
    full = vocoder(args.model, "float32")
    # the decoder draws its noise buffer at construction, so the second load gets other noise
    full.flow.decoder.rand_noise = half.flow.decoder.rand_noise.clone()
    noise_sum = float(half.flow.decoder.rand_noise.double().sum())
    print(json.dumps({"rand_noise_sum": noise_sum}), flush=True)
    saved = {}
    results = []
    for name, token_counts in CASES:
        generator = torch.Generator().manual_seed(sum(token_counts))
        items = [
            FlowBatchInput(
                token=torch.randint(0, 6561, (1, tokens), generator=generator),
                prompt_token=torch.randint(
                    0, 6561, (1, PROMPT_TOKENS), generator=generator
                ),
                prompt_feat=torch.randn(
                    1, PROMPT_TOKENS * 2, half.flow.output_size, generator=generator
                ),
                embedding=torch.randn(
                    1, half.flow.spk_embed_affine_layer.in_features, generator=generator
                ),
            )
            for tokens in token_counts
        ]
        with torch.inference_mode():
            with half.stream_context:
                bf16_mels = half.leftover_batch(items)
            with full.stream_context:
                float32_mels = full.leftover_batch(items)
        errors = [
            float(
                torch.linalg.vector_norm(a.float() - b.float())
                / torch.linalg.vector_norm(b.float())
            )
            for a, b in zip(bf16_mels, float32_mels, strict=True)
        ]
        row = {
            "case": name,
            "relative_error_per_row": errors,
            "float32_norm_per_row": [float(m.float().norm()) for m in float32_mels],
        }
        print(json.dumps(row), flush=True)
        results.append(row)
        saved[name] = {
            "bfloat16": [m.float().cpu() for m in bf16_mels],
            "float32": [m.float().cpu() for m in float32_mels],
        }
    torch.save(saved, args.out.replace(".json", ".pt"))
    with open(args.out, "w") as handle:
        json.dump({"rand_noise_sum": noise_sum, "cases": results}, handle, indent=1)


if __name__ == "__main__":
    main()
