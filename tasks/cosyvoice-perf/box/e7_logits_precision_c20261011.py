# SPDX-License-Identifier: Apache-2.0
"""E7: what the speech head's precision does to the distribution the AR samples from (item 1).

CosyVoice's own CosyVoice3 (float32, the reference) speaks SeedTTS English zero shot prompts while a hook
keeps the input of every decode step's `llm_decoder` call (the last hidden state). For each kept state h and
the head weight W, against the float64 reference log_softmax(h W^T / T):
  bf16      omni today: h and W in bf16, bf16 logits, temperature and softmax in place in bf16 (SGLang's
            sampler, layers/sampler.py:248, 266, given bf16 logits)
  upcast    SGLang's default head: bf16 logits made float32 (logits_processor.py:1158), float32 softmax
  fp32_out  SGLang's fp32 head: torch.mm(h_bf16, W_bf16^T, out_dtype=float32) (logits_processor.py:965-985)
Per variant and temperature (1.0, CosyVoice's, and 0.7, omni's): mean and max KL(reference || variant), max
absolute probability error, and the share of steps whose top k 20 / top p 0.8 kept set differs from the
reference's (SGLang's rule, sampler.py:766-772).

    python e7_logits_precision_c20261011.py --out e7.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3 import stages

PROMPTS = 8
TOP_K = 20
TOP_P = 0.8


def kept_set(probs: torch.Tensor) -> torch.Tensor:
    """SGLang's torch top k then top p mask over sorted probabilities, as a boolean vocabulary mask."""
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    sorted_probs[..., TOP_K:] = 0.0
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs[(cumulative - sorted_probs) > TOP_P] = 0.0
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(-1, order, sorted_probs > 0)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from cosyvoice.cli.cosyvoice import CosyVoice3

    model = CosyVoice3(stages.resolve_checkpoint("FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))
    # note(ratish): the installed transformers builds the Qwen2 backbone in the checkpoint config's bf16,
    # CosyVoice3's default (fp16=False) runs the whole LM in float32.
    model.model.llm.float()
    decoder = model.model.llm.llm_decoder
    hidden: list[torch.Tensor] = []
    decoder.register_forward_hook(lambda module, inputs, output: hidden.append(inputs[0].detach()))
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")[:PROMPTS]
    for sample in samples:
        # note(ratish): CosyVoice's frontend loads and resamples the reference itself.
        for _ in model.inference_zero_shot(
            sample.target_text,
            "You are a helpful assistant.<|endofprompt|>" + sample.ref_text,
            sample.ref_audio,
            stream=False,
        ):
            pass
    states = torch.cat([h.reshape(-1, h.shape[-1]) for h in hidden], dim=0).float()
    weight = decoder.weight.detach().float()
    bias = decoder.bias.detach().float() if decoder.bias is not None else None
    print(f"decode steps kept {states.shape[0]}, head {tuple(weight.shape)}", flush=True)
    report: dict[str, dict] = {"steps": {"count": int(states.shape[0])}}
    state_bf16, weight_bf16 = states.bfloat16(), weight.bfloat16()
    for temperature in (1.0, 0.7):
        reference_logits = states.double() @ weight.double().T
        if bias is not None:
            reference_logits = reference_logits + bias.double()
        reference = torch.log_softmax(reference_logits / temperature, dim=-1)
        reference_probs = reference.exp()
        reference_kept = kept_set(reference_probs.float())
        variants = {}
        bf16_logits = state_bf16 @ weight_bf16.T
        variants["bf16"] = torch.softmax(bf16_logits.div(temperature), dim=-1).double()
        variants["upcast"] = torch.softmax(bf16_logits.float().div(temperature), dim=-1).double()
        fp32_out = torch.mm(state_bf16, weight_bf16.T, out_dtype=torch.float32)
        variants["fp32_out"] = torch.softmax(fp32_out.div(temperature), dim=-1).double()
        for name, probs in variants.items():
            kl = (reference_probs * (reference - probs.clamp_min(1e-300).log())).sum(dim=-1)
            kept_differs = (kept_set(probs.float()) != reference_kept).any(dim=-1)
            report[f"T={temperature} {name}"] = {
                "kl_mean": float(kl.mean()),
                "kl_max": float(kl.max()),
                "prob_abs_err_max": float((probs - reference_probs).abs().max()),
                "kept_set_differs_share": float(kept_differs.float().mean()),
            }
            print(f"T={temperature} {name}", report[f"T={temperature} {name}"], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
