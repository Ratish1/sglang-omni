# SPDX-License-Identifier: Apache-2.0
"""Compiled PackedDiT against its own eager forward on real weights, per served shape.

The vocoder is built by its own factory (compile on); each case runs the compiled hop or
final, then the same call with the instance's compiled forward removed (the class's
eager forward), and reports bit equality, max abs and relative L2 of the mel. A last
pass runs every case eager in float32 (the same bf16-rounded weights, autocast off) as
the arithmetic reference both bf16 paths are measured against.

usage: cd <tree> && python packed_drift_probe.py --out <json>
"""

from __future__ import annotations

import argparse
import json

import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    FlowBatchInput,
    create_vocoder_executor,
)

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
PROMPT_TOKENS = 75
CASES = {
    "hop1": ("hop", 1, 28),
    "hop16": ("hop", 16, 28),
    "hop16_long": ("hop", 16, 178),
    "final1": ("final", 1, 100),
    "final16": ("final", 16, 100),
}


def make_items(flow, rows: int, tokens: int) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(rows * 1000 + tokens)
    return [
        FlowBatchInput(
            token=torch.randint(0, 6561, (1, tokens), generator=generator),
            prompt_token=torch.randint(
                0, 6561, (1, PROMPT_TOKENS), generator=generator
            ),
            prompt_feat=torch.randn(
                1, PROMPT_TOKENS * 2, flow.output_size, generator=generator
            )
            * 2
            - 5,
            embedding=torch.randn(
                1, flow.spk_embed_affine_layer.in_features, generator=generator
            ),
        )
        for _ in range(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scheduler = create_vocoder_executor(
        MODEL,
        device="cuda",
        gpu_id=0,
        enable_flow_cuda_graph=False,
    )
    vocoder = scheduler.vocoder
    estimator = vocoder.flow.packed_estimator
    compiled_forward = estimator.forward
    results = {}
    kept = {}
    with torch.inference_mode(), vocoder.stream_context:
        for name, (kind, rows, tokens) in CASES.items():
            items = make_items(vocoder.flow, rows, tokens)
            call = vocoder.hop_batch if kind == "hop" else vocoder.leftover_batch
            estimator.forward = compiled_forward
            compiled = torch.cat([mel.float() for mel in call(items)], dim=2)
            del estimator.forward
            eager = torch.cat([mel.float() for mel in call(items)], dim=2)
            estimator.forward = compiled_forward
            kept[name] = (items, call, compiled.cpu(), eager.cpu())
            results[name] = dict(
                equal=bool(torch.equal(compiled, eager)),
                max_abs=float((compiled - eager).abs().max()),
                relative_l2=float(
                    torch.linalg.vector_norm(compiled - eager)
                    / torch.linalg.vector_norm(eager)
                ),
            )
            print(name, results[name], flush=True)
    del estimator.forward
    estimator.dit.float()
    vocoder.autocast_dtype = None
    with torch.inference_mode(), vocoder.stream_context:
        for name, (items, call, compiled, eager) in kept.items():
            truth = torch.cat([mel.float() for mel in call(items)], dim=2).cpu()
            norm = torch.linalg.vector_norm(truth)
            results[name]["compiled_vs_fp32"] = float(
                torch.linalg.vector_norm(compiled - truth) / norm
            )
            results[name]["eager_vs_fp32"] = float(
                torch.linalg.vector_norm(eager - truth) / norm
            )
            print(name, "vs fp32", results[name], flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
