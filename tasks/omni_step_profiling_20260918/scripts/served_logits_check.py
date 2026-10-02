"""The served predictor's own logits: the sampler's settings and calibration, whether the
offline harness reproduces them, and how far each arm's served distributions sit from fp32.

Inputs: record_predictor_chain files written with PREDICTOR_CHAIN_LOGITS=1 by eager boots
(SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0; the same kernels the graphs capture), one per arm,
given as LABEL:ARM=PATH with ARM plain (main) or fused (P4). Per file:

  1  the sampling buffers the sampler read (temperature, top k, top p, do sample)
  2  sampler calibration on the served logits: codes outside the served top k, and the mean
     log probability of the sampled codes against the mean negative entropy of the served
     distributions (equal in expectation for a correct sampler), with standard errors
  3  reproduction: every served call rebuilt at its exact batch with the arm's eager path
     (predictor_teacher_forced fixtures and weights, served projected tables), logits
     compared with the served logits bit for bit
  4  distance to fp32, teacher forced along the served codes: per codebook KL of the served
     softmax at the served temperature from the fp32 one, total variation of the top k
     sampling distributions, entropy shift, and the fp32 log probability of the served codes

usage: PYTHONPATH=<P4 tree>:<this dir> python served_logits_check.py main:plain=A.pt
       p4:fused=B.pt [--calls 400]
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from predictor_teacher_forced import DTYPE, Fp32Chain, fixtures, load
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context

import sglang_omni.models.qwen3_tts.sglang_model as sglang_model_module
from sglang_omni.vendor.sglang.models import apply_qk_norm


class Mean:
    def __init__(self) -> None:
        self.n, self.total, self.squares = 0, 0.0, 0.0

    def add(self, values: torch.Tensor) -> None:
        values = values.double().flatten()
        self.n += values.numel()
        self.total += float(values.sum())
        self.squares += float((values**2).sum())

    def text(self, digits: int = 5) -> str:
        mean = self.total / self.n
        se = math.sqrt(max(self.squares / self.n - mean**2, 0.0) / self.n)
        return f"{mean:.{digits}f} (se {se:.{digits}f})"


def topk_distribution(logits, temperature, top_k):
    scores = logits.float() / temperature.view(-1, 1)
    values, indices = torch.topk(scores, top_k, dim=-1)
    probs = torch.zeros_like(scores)
    probs.scatter_(1, indices, torch.softmax(values, dim=-1))
    return probs


def entropy(probs):
    return -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)


def chain_steps(data, rows, chain, tables, device):
    b = rows.stop - rows.start
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
    for j in range(len(tables)):
        steps.append(F.embedding(codes[:, j], tables[j]).view(b, 1, -1))
    return steps, codes


def run_arm(talker, steps, heads, b):
    talker.predictor_k_cache.zero_()
    talker.predictor_v_cache.zero_()
    logits, cache_len = [], 0
    for j, step in enumerate(steps):
        out = talker.predictor_forward_tokens(
            token_embeds=step.clone(), batch_size=b, cache_len=cache_len
        )
        last = out[:, 1] if j == 0 else out[:, 0]
        logits.append(F.linear(last, heads[j]))
        cache_len += step.shape[1]
    return torch.stack(logits, 1)


def run_fp32(layers, final_norm, eps, steps, heads, b, device):
    reference = Fp32Chain(layers, final_norm, eps, b, device)
    logits, cache_len = [], 0
    for j, step in enumerate(steps):
        out = reference.forward(step, cache_len)
        last = out[:, 1] if j == 0 else out[:, 0]
        logits.append(last @ heads[j].float().t())
        cache_len += step.shape[1]
    return torch.stack(logits, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--calls", type=int, default=400)
    args = parser.parse_args()
    device = torch.device("cuda")
    sglang_model_module.apply_qk_norm = apply_qk_norm
    layers, final_norm, eps, chain = load(device)
    heads = chain["lm_heads"]
    tables = [
        F.linear(weight, chain["projection_weight"], chain["projection_bias"])
        for weight in chain["codec_embeddings"][: len(heads) - 1]
    ]
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
    for item in args.files:
        label_arm, path = item.split("=", 1)
        label, arm = label_arm.split(":")
        data = torch.load(path)
        rows_total = data["layer0_codes"].shape[0]
        print(f"\n===== {label} ({arm} path), {rows_total} served rows")
        for key in ("temperature", "top_k", "top_p", "do_sample"):
            values, counts = torch.unique(data[key].float(), return_counts=True)
            print(
                f"1 {key}: "
                + ", ".join(
                    f"{v:g} x{c}" for v, c in zip(values.tolist(), counts.tolist())
                )
            )
        served_logits = data["logits"].to(device)
        codes = data["codes"].to(device)
        temperature = data["temperature"].to(device)
        top_k = int(data["top_k"].max())
        outside = 0
        logq, neg_h = Mean(), Mean()
        for j in range(served_logits.shape[1]):
            q = topk_distribution(served_logits[:, j], temperature, top_k)
            p_code = q.gather(1, codes[:, j : j + 1]).view(-1)
            outside += int((p_code == 0).sum())
            logq.add(torch.log(p_code[p_code > 0]))
            neg_h.add(-entropy(q))
        print(
            f"2 served codes outside the served top-{top_k}: {outside}; mean log q(code) "
            f"{logq.text()} against mean -H(q) {neg_h.text()}"
        )
        calls = []
        index = 0
        while index < rows_total and len(calls) < args.calls:
            b = int(data["batch_size"][index])
            calls.append(slice(index, index + b))
            index += b
        equal_rows = total_rows = 0
        max_diff = 0.0
        kl = [Mean() for _ in heads]
        tv = [Mean() for _ in heads]
        dh = [Mean() for _ in heads]
        logp = [Mean() for _ in heads]
        with (
            get_context().override_server_args(
                cuda_graph_config=CudaGraphConfig(
                    prefill=PhaseConfig(backend=Backend.DISABLED)
                )
            ),
            torch.no_grad(),
        ):
            for rows in calls:
                b = rows.stop - rows.start
                steps, call_codes = chain_steps(data, rows, chain, tables, device)
                rebuilt = run_arm(talkers[arm], steps, heads, b)
                served = served_logits[rows]
                same = (rebuilt == served).reshape(b, -1).all(1)
                equal_rows += int(same.sum())
                total_rows += b
                max_diff = max(
                    max_diff, float((rebuilt.float() - served.float()).abs().max())
                )
                reference = run_fp32(layers, final_norm, eps, steps, heads, b, device)
                temp = temperature[rows]
                for j in range(len(heads)):
                    log_q = F.log_softmax(served[:, j].float() / temp.view(-1, 1), -1)
                    log_p = F.log_softmax(reference[:, j] / temp.view(-1, 1), -1)
                    kl[j].add((log_q.exp() * (log_q - log_p)).sum(-1))
                    q = topk_distribution(served[:, j], temp, top_k)
                    p = topk_distribution(reference[:, j], temp, top_k)
                    tv[j].add(0.5 * (q - p).abs().sum(-1))
                    dh[j].add(entropy(q) - entropy(p))
                    p_code = p.gather(1, call_codes[:, j : j + 1]).view(-1)
                    logp[j].add(torch.log(p_code.clamp_min(1e-30)))
        print(
            f"3 reproduction over {len(calls)} calls ({total_rows} rows): rows bit equal "
            f"{equal_rows} of {total_rows}, max abs logit diff {max_diff:.4g}"
        )
        print(
            "4 per codebook against fp32: KL(served || fp32) at the served temperature, TV of the top-k distributions, entropy shift, fp32 log p of the served code"
        )
        all_kl, all_tv, all_dh, all_lp = Mean(), Mean(), Mean(), Mean()
        for j in range(len(heads)):
            print(
                f"  code {j + 1:>2}: KL {kl[j].text()}  TV {tv[j].text()}  dH {dh[j].text()}  log p {logp[j].text(4)}"
            )
            for total, part in (
                (all_kl, kl[j]),
                (all_tv, tv[j]),
                (all_dh, dh[j]),
                (all_lp, logp[j]),
            ):
                total.n += part.n
                total.total += part.total
                total.squares += part.squares
        print(
            f"  all     : KL {all_kl.text()}  TV {all_tv.text()}  dH {all_dh.text()}  log p {all_lp.text(4)}"
        )


if __name__ == "__main__":
    main()
