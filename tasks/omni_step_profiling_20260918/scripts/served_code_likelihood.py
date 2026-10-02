"""How likely the codes a server actually sampled are under the fp32 predictor.

Inputs: record_predictor_chain files from served boots (each holds what that server's
predictor really sampled, through its CUDA graphs, bucket padding, projected embedding
gather and seeded sampler). For each file, teacher forced along the served codes, the
fp32 reference (the checkpoint's bf16 weights in fp32, predictor_teacher_forced.Fp32Chain)
gives every pass's sampling distribution at the served temperature and top k; the served
code's log probability under it is read per codebook, with the share of served codes
outside the fp32 top k. A server that samples from distributions further from fp32 shows a
lower mean log probability. The plain and fused bf16 arms run eagerly alongside: every
served code must lie in its own arm's eager top k (a code outside would mean the served
path computed other logits than the eager launches), and their log probabilities of the
served codes are printed too.

usage: PYTHONPATH=<P4 tree> python served_code_likelihood.py LABEL=CHAIN.pt ... [--rows N]
       [--batch 16] [--temperature 0.9] [--top-k 50]
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from predictor_teacher_forced import (
    DTYPE,
    Fp32Chain,
    fixtures,
    load,
    sampling_distribution,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.vendor.sglang.models import apply_qk_norm


def score(path, layers, final_norm, eps, chain, talkers, args, device):
    data = torch.load(path)
    # the served passes read rows of the projected tables built over whole
    # codebooks (Qwen3TTSTalker.post_load_weights), so the inputs here do the same.
    tables = [
        F.linear(weight, chain["projection_weight"], chain["projection_bias"])
        for weight in chain["codec_embeddings"][: len(chain["lm_heads"]) - 1]
    ]
    total = min(args.rows, data["layer0_codes"].shape[0]) // args.batch * args.batch
    codes_per_pass = len(chain["lm_heads"])
    sums = {
        name: torch.zeros(codes_per_pass, dtype=torch.float64)
        for name in ("fp32", "plain", "fused")
    }
    squares = torch.zeros(codes_per_pass, dtype=torch.float64)
    outside = {
        name: torch.zeros(codes_per_pass, dtype=torch.long)
        for name in ("fp32", "plain", "fused")
    }
    for start in range(0, total, args.batch):
        rows = slice(start, start + args.batch)
        b = args.batch
        hidden = data["talker_hidden"][rows].to(device, DTYPE).view(b, 1, -1)
        code0 = data["layer0_codes"][rows].to(device)
        codes = data["codes"][rows].to(device)
        embed0 = F.embedding(code0, chain["layer0_embedding"]).view(b, 1, -1)
        steps = [
            F.linear(
                torch.cat((hidden, embed0), 1),
                chain["projection_weight"],
                chain["projection_bias"],
            )
        ]
        for j in range(codes_per_pass - 1):
            steps.append(F.embedding(codes[:, j], tables[j]).view(b, 1, -1))
        for talker in talkers.values():
            talker.predictor_k_cache.zero_()
            talker.predictor_v_cache.zero_()
        reference = Fp32Chain(layers, final_norm, eps, b, device)
        cache_len = 0
        for j, step in enumerate(steps):
            last = slice(1, 2) if j == 0 else slice(0, 1)
            head = chain["lm_heads"][j]
            served = codes[:, j]
            ref_out = reference.forward(step, cache_len)
            outs = {"fp32": ref_out[:, last].reshape(b, -1) @ head.float().t()}
            for name, talker in talkers.items():
                hidden_out = talker.predictor_forward_tokens(
                    token_embeds=step.clone(), batch_size=b, cache_len=cache_len
                )
                outs[name] = F.linear(hidden_out[:, last].reshape(b, -1), head)
            for name, logits in outs.items():
                probs = sampling_distribution(logits, args.temperature, args.top_k)
                p = probs.gather(1, served.view(-1, 1)).view(-1).double()
                outside[name][j] += int((p == 0).sum())
                logp = torch.log(p.clamp_min(1e-30))
                sums[name][j] += float(logp[p > 0].sum())
                if name == "fp32":
                    squares[j] += float((logp[p > 0] ** 2).sum())
                else:
                    pass
            cache_len += step.shape[1]
    return total, sums, squares, outside


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chains", nargs="+")
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda")
    sglang_model_module.apply_qk_norm = apply_qk_norm
    layers, final_norm, eps, chain = load(device)
    talkers = {
        name: fixtures.real_shape_talker(device, layers, final_norm, fused=fused)
        for name, fused in (("plain", False), ("fused", True))
    }
    plain = talkers["plain"]
    plain.predictor_k_rows = [
        c.view(-1, c.shape[-2] * c.shape[-1]) for c in plain.predictor_k_cache
    ]
    plain.predictor_v_rows = [
        c.view(-1, c.shape[-2] * c.shape[-1]) for c in plain.predictor_v_cache
    ]
    plain.predictor_rope_stores_kv = True
    results = {}
    with (
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            )
        ),
        torch.no_grad(),
    ):
        for item in args.chains:
            label, path = item.split("=", 1)
            results[label] = score(
                path, layers, final_norm, eps, chain, talkers, args, device
            )
    print(
        "mean log probability of the served codes under each arm's sampling distribution "
        f"(temperature {args.temperature}, top-k {args.top_k}); outside = served codes not in "
        "that arm's top-k"
    )
    for label, (rows, sums, squares, outside) in results.items():
        print(f"\n{label}: {rows} served rows")
        print(
            f"{'code':>4} {'fp32 logp':>10} {'plain logp':>11} {'fused logp':>11} {'outside fp32':>13} {'outside plain':>14} {'outside fused':>14}"
        )
        for j in range(len(sums["fp32"])):
            n = rows - int(outside["fp32"][j])
            print(
                f"{j + 1:>4} {sums['fp32'][j] / n:>10.4f} {sums['plain'][j] / (rows - int(outside['plain'][j])):>11.4f} "
                f"{sums['fused'][j] / (rows - int(outside['fused'][j])):>11.4f} {int(outside['fp32'][j]):>13} "
                f"{int(outside['plain'][j]):>14} {int(outside['fused'][j]):>14}"
            )
        count = rows * len(sums["fp32"]) - int(outside["fp32"].sum())
        mean = float(sums["fp32"].sum()) / count
        variance = float(squares.sum()) / count - mean**2
        print(
            f"all  fp32 logp {mean:.4f} (se {math.sqrt(variance / count):.4f}, n {count}); "
            f"outside fp32 {int(outside['fp32'].sum())}, plain {int(outside['plain'].sum())}, "
            f"fused {int(outside['fused'].sum())}"
        )


if __name__ == "__main__":
    main()
