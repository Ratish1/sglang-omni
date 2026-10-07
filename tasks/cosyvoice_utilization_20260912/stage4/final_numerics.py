# SPDX-License-Identifier: Apache-2.0
"""A tree's final (leftover) Flow mels over fixed inputs, one vocoder per process (a second
vocoder in one process gave broken multi-row finals), graphs off. Runs of one tree at other
precisions or with the DiT compile, and runs of another tree, are compared with --compare.
The decoder noise is drawn at construction from the process's default seed, so every run
starts its ODE from the same noise (its sum is printed to check).

    cd <tree> && python final_numerics.py --dtype bfloat16 [--compile] --out <run>.pt
    python final_numerics.py --compare <a>.pt <b>.pt
"""

from __future__ import annotations

import argparse
import json

import torch

CASES = (("1 row", (100,)), ("4 rows", (60, 100, 140, 180)), ("1 long row", (480,)))
PROMPT_TOKENS = 75


def run(args: argparse.Namespace) -> None:
    from sglang_omni.models.fun_cosyvoice3 import stages
    from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = dict(
        enable_flow_prefix_cuda_graph=False,
        flow_prefix_cache_gb=0.0,
        enable_dit_torch_compile=args.compile,
        enable_flow_cuda_graph=False,
    )
    if (
        "enable_flow_final_cuda_graph"
        in stages.create_vocoder_executor.__code__.co_varnames
    ):
        kwargs["enable_flow_final_cuda_graph"] = False
    else:
        pass
    vocoder = stages.create_vocoder_executor(
        args.model, device="cuda", gpu_id=0, dtype=args.dtype, **kwargs
    ).vocoder
    saved = {"rand_noise_sum": float(vocoder.flow.decoder.rand_noise.double().sum())}
    for name, token_counts in CASES:
        generator = torch.Generator().manual_seed(sum(token_counts))
        items = [
            FlowBatchInput(
                token=torch.randint(0, 6561, (1, tokens), generator=generator),
                prompt_token=torch.randint(
                    0, 6561, (1, PROMPT_TOKENS), generator=generator
                ),
                prompt_feat=torch.randn(
                    1, PROMPT_TOKENS * 2, vocoder.flow.output_size, generator=generator
                ),
                embedding=torch.randn(
                    1,
                    vocoder.flow.spk_embed_affine_layer.in_features,
                    generator=generator,
                ),
            )
            for tokens in token_counts
        ]
        with torch.inference_mode(), vocoder.stream_context:
            saved[name] = [mel.float().cpu() for mel in vocoder.leftover_batch(items)]
    print(
        json.dumps(
            {
                "rand_noise_sum": saved["rand_noise_sum"],
                "norms": {
                    name: [float(mel.norm()) for mel in saved[name]]
                    for name, _ in CASES
                },
            }
        ),
        flush=True,
    )
    torch.save(saved, args.out)


def compare(first_path: str, second_path: str) -> None:
    first = torch.load(first_path)
    second = torch.load(second_path)
    assert (
        first["rand_noise_sum"] == second["rand_noise_sum"]
    ), "different decoder noise"
    for name, _ in CASES:
        print(
            json.dumps(
                {
                    "case": name,
                    "relative_difference_per_row": [
                        float((a - b).norm() / b.norm())
                        for a, b in zip(first[name], second[name], strict=True)
                    ],
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
