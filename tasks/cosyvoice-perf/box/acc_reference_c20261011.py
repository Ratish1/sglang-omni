# SPDX-License-Identifier: Apache-2.0
"""Component accuracy of omni's Fun-CosyVoice3 serving path against CosyVoice's own modules (074ca6dc), from
the records of acc_probe/sitecustomize.py (one served seeded c1 cell per tree).

Truth is CosyVoice's module in float64. Each omni component is compared on its own inputs as served, beside
CosyVoice's module in the dtypes that bound what precision alone does:
  frontend  CosyVoice's frontend_zero_shot on the same SeedTTS sample against omni's prepared request: text
            ids, the LM's and the Flow's prompt speech tokens, the prompt mel, the speaker embedding (equal or
            max difference), and whether CosyVoice's text normalization would change the text.
  LM        teacher forced on omni's sampled ids: omni's served logits and CosyVoice's LM in float32 (the
            checkpoint's weights, as CosyVoice's pinned transformers 4.51.3 builds it), float32 over bf16
            rounded backbone weights (what transformers 5 builds, E7's reference), bf16 with the float32 head
            output (item 1) and bf16 throughout (main), each against float64: KL, top 1, the kept sets of
            omni's rule (top k 20, top p 0.8, temperature 0.7) and CosyVoice's (top k 25, top p 0.8,
            temperature 1), the largest logit error. Also omni's prompt embeddings against CosyVoice's.
  Flow      every served step on omni's tokens and padded prompt: omni's new mel frames and CosyVoice's Flow
            in float32 and under bf16 autocast, against float64 (SNR in dB, largest error).
  HiFT      every served step on omni's own mel history: omni's emitted samples and CosyVoice's HiFT in
            float32, against float64.
  chain     CosyVoice's Flow then HiFT in float32 on omni's tokens with omni's padded prompt and step bounds
            (numerics of the whole vocoder), and with CosyVoice's own prompt and hop schedule (the semantics,
            ARCH_AUDIT 1.1), against omni's audio.

    python3 acc_reference_c20261011.py --records <probe dir> --out <json>
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import statistics

import torch

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fun_cosyvoice3 import stages

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
PRE_LOOKAHEAD = 3
HOP = 25
MAX_HOP = 100
RULES = {"omni k20 p0.8 T0.7": (20, 0.8, 0.7), "cosyvoice k25 p0.8 T1": (25, 0.8, 1.0)}


def kept_set(probs: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Top k then top p over sorted probabilities (SGLang's rule and CosyVoice's nucleus_sampling)."""
    sorted_probs, order = probs.sort(dim=-1, descending=True, stable=True)
    sorted_probs[..., top_k:] = 0.0
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs[(cumulative - sorted_probs) >= top_p] = 0.0
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(-1, order, sorted_probs > 0)
    return mask


def snr_db(reference: torch.Tensor, value: torch.Tensor) -> float:
    reference, value = reference.double().flatten().cpu(), value.double().flatten().cpu()
    if reference.shape != value.shape:
        return float("nan")
    noise = (reference - value).pow(2).sum()
    return float(10 * torch.log10(reference.pow(2).sum() / noise)) if noise > 0 else math.inf


def max_err(reference: torch.Tensor, value: torch.Tensor) -> float:
    reference, value = reference.double().flatten().cpu(), value.double().flatten().cpu()
    return float((reference - value).abs().max()) if reference.shape == value.shape else float("nan")


def summary(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"n": len(values)}
    return {"n": len(values), "min": min(finite), "median": statistics.median(finite), "max": max(finite)}


def build_lm(base, state: dict, device: str, dtype: torch.dtype, round_backbone: bool = False):
    lm = copy.deepcopy(base).to(device=device, dtype=dtype)
    lm.load_state_dict(state, strict=True)
    if round_backbone:
        for parameter in lm.llm.parameters():
            parameter.data = parameter.data.bfloat16().to(dtype)
    return lm.eval()


@torch.inference_mode()
def lm_logits(lm, record: dict, device: str, fp32_head: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """CosyVoice3LM.inference's lm_input, then one causal pass over it and omni's sampled ids (teacher
    forcing), the head over the position before each sampled id."""
    text = torch.tensor([record["text_token_ids"]], device=device)
    text_emb = lm.llm.model.model.embed_tokens(text)
    sos = lm.speech_embedding.weight[lm.sos].reshape(1, 1, -1)
    task = lm.speech_embedding.weight[lm.task_id].reshape(1, 1, -1)
    parts = [sos, text_emb, task]
    if record["llm_prompt_speech_token_ids"]:
        parts.append(lm.speech_embedding(torch.tensor([record["llm_prompt_speech_token_ids"]], device=device)))
    lm_input = torch.cat(parts, dim=1)
    sampled = record["sampled"]
    sequence = lm_input
    if len(sampled) > 1:
        sequence = torch.cat([lm_input, lm.speech_embedding(torch.tensor([sampled[:-1]], device=device))], dim=1)
    # note(ratish): Qwen2Encoder.forward reads hidden_states[-1], the final norm's output under the pinned
    # transformers 4.51.3; last_hidden_state is that output under any version.
    hidden = lm.llm.model.model(inputs_embeds=sequence).last_hidden_state
    start = lm_input.shape[1] - 1
    states = hidden[0, start : start + len(sampled)]
    if fp32_head:
        logits = torch.mm(states, lm.llm_decoder.weight.T, out_dtype=torch.float32)
    else:
        logits = lm.llm_decoder(states)
    return lm_input[0], logits


def probabilities(logits: torch.Tensor, temperature: float, sampler_dtype: torch.dtype) -> torch.Tensor:
    """The sampler's softmax in the dtype the logits reach it in (SGLang sampler.py, in place)."""
    return torch.softmax(logits.to(sampler_dtype).div(temperature), dim=-1).double()


def lm_metrics(reference: torch.Tensor, logits: torch.Tensor, sampler_dtype: torch.dtype) -> dict[str, float]:
    reference = reference.double()
    out = {
        "logit_err_max": float((logits.double() - reference).abs().max()),
        "top1_agree": float((logits.double().argmax(-1) == reference.argmax(-1)).double().mean()),
    }
    for temperature in (1.0, 0.7):
        reference_log = torch.log_softmax(reference / temperature, dim=-1)
        probs = probabilities(logits, temperature, sampler_dtype)
        kl = (reference_log.exp() * (reference_log - probs.clamp_min(1e-300).log())).sum(-1)
        out[f"kl_mean_T{temperature}"] = float(kl.mean())
        out[f"kl_max_T{temperature}"] = float(kl.max())
        out[f"prob_err_max_T{temperature}"] = float((probs - reference_log.exp()).abs().max())
    for name, (top_k, top_p, temperature) in RULES.items():
        reference_probs = torch.softmax(reference / temperature, dim=-1)
        probs = probabilities(logits, temperature, sampler_dtype)
        differs = (kept_set(probs, top_k, top_p) != kept_set(reference_probs, top_k, top_p)).any(-1)
        out[f"kept_differs {name}"] = float(differs.double().mean())
    return out


def lm_report(cosy, root: str, ar_records: dict, device: str) -> dict:
    state = torch.load(os.path.join(root, "llm.pt"), map_location="cpu", weights_only=True)
    base = cosy.model.llm
    variants = {
        "cosyvoice fp32": (build_lm(base, state, device, torch.float32), False, torch.float32),
        "cosyvoice fp32, bf16 backbone weights (E7)": (
            build_lm(base, state, device, torch.float32, round_backbone=True),
            False,
            torch.float32,
        ),
        "bf16, fp32 head out (item 1)": (build_lm(base, state, device, torch.bfloat16), True, torch.float32),
        "bf16 (main)": (build_lm(base, state, device, torch.bfloat16), False, torch.bfloat16),
    }
    reference_lm = build_lm(base, state, device, torch.float64)
    steps: dict[str, list[dict]] = {}
    embeds = []
    for request_id, record in ar_records.items():
        served = torch.stack(record["logits"]).to(device)
        if served.shape[0] != len(record["sampled"]):
            print(f"skip {request_id}: {served.shape[0]} logits rows, {len(record['sampled'])} sampled", flush=True)
            continue
        lm_input, reference = lm_logits(reference_lm, record, device, fp32_head=False)
        _, cosy32 = lm_logits(variants["cosyvoice fp32"][0], record, device, fp32_head=False)
        omni_embeds = record["prompt_input_embeds"].to(device)
        embeds.append(
            {
                "rows": int(omni_embeds.shape[0]),
                "rows_expected": int(lm_input.shape[0]),
                "equal_to_cosyvoice_bf16": bool(
                    omni_embeds.shape == lm_input.shape
                    and torch.equal(omni_embeds.to(torch.bfloat16), lm_input.to(torch.float32).bfloat16())
                ),
                "max_err_vs_fp64": max_err(lm_input, omni_embeds) if omni_embeds.shape == lm_input.shape else None,
            }
        )
        # note(ratish): the served logits keep their own dtype's values (main's bf16 logits were stored as float).
        served_dtype = torch.bfloat16 if torch.equal(served, served.bfloat16().float()) else torch.float32
        steps.setdefault("omni served", []).append(
            {"request": request_id, "steps": int(served.shape[0]), **lm_metrics(reference, served, served_dtype)}
        )
        for name, (lm, fp32_head, sampler_dtype) in variants.items():
            if lm is variants["cosyvoice fp32"][0]:
                logits = cosy32
            else:
                _, logits = lm_logits(lm, record, device, fp32_head=fp32_head)
            steps.setdefault(name, []).append(
                {"request": request_id, "steps": int(logits.shape[0]), **lm_metrics(reference, logits, sampler_dtype)}
            )
        print(f"lm {request_id} steps {served.shape[0]} served dtype {served_dtype}", flush=True)
    report = {"prompt_embeddings": embeds, "variants": {}}
    for name, rows in steps.items():
        total = sum(row["steps"] for row in rows)
        merged = {"requests": len(rows), "steps": total}
        for key in rows[0]:
            if key in ("request", "steps"):
                continue
            weights = [row["steps"] for row in rows]
            values = [row[key] for row in rows]
            if key.startswith(("kl_max", "logit_err_max", "prob_err_max")):
                merged[key] = max(values)
            else:
                merged[key] = sum(value * weight for value, weight in zip(values, weights)) / total
        report["variants"][name] = merged
    return report


@torch.inference_mode()
def flow_mel(flow, tokens: list[int], prompt_token, prompt_feat, embedding, finalize: bool, device, dtype, autocast):
    token = torch.tensor([tokens], dtype=torch.int32, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        feat, _ = flow.inference(
            token=token,
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32, device=device),
            prompt_token=prompt_token.to(device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32, device=device),
            prompt_feat=prompt_feat.to(device=device, dtype=dtype),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=device),
            embedding=embedding.to(device=device, dtype=dtype),
            # note(ratish): CosyVoice's streaming loop runs its hops with the chunk mask and its final with
            # stream left at its default False, full attention (cli/model.py:367-373, 425, 434).
            streaming=not finalize,
            finalize=finalize,
        )
    return feat


@torch.inference_mode()
def hift_samples(hift, history: torch.Tensor, finalize: bool, device, dtype) -> torch.Tensor:
    speech, _ = hift.inference(speech_feat=history.to(device=device, dtype=dtype), finalize=finalize)
    return speech


def vocoder_report(cosy, ar_records: dict, voc_records: dict, device: str) -> dict:
    flow32, hift32 = cosy.model.flow, cosy.model.hift
    flow64 = copy.deepcopy(flow32).double()
    hift64 = copy.deepcopy(hift32).double()
    rows = {key: [] for key in ("flow omni", "flow cosyvoice bf16 autocast", "flow cosyvoice fp32",
                                "hift omni", "hift cosyvoice fp32")}
    chains = []
    for request_id, record in voc_records.items():
        tokens, steps = record["tokens"], record["steps"]
        prompt_token, prompt_feat, embedding = record["prompt_token"], record["prompt_feat"], record["embedding"]
        history_omni = None
        chain = {"request": request_id, "steps": len(steps), "tokens": len(tokens)}
        chain_mel, chain_audio, omni_audio = None, [], []
        for step in steps:
            offset, final = step["token_offset"], step["is_final"]
            end = len(tokens) if final else offset + step["hop_len"] + PRE_LOOKAHEAD
            used = tokens[:end]
            frames = slice(offset * 2, None)
            mel64 = flow_mel(flow64, used, prompt_token, prompt_feat, embedding, final, device, torch.float64, False)
            mel32 = flow_mel(flow32, used, prompt_token, prompt_feat, embedding, final, device, torch.float32, False)
            mel16 = flow_mel(flow32, used, prompt_token, prompt_feat, embedding, final, device, torch.float32, True)
            new_omni = step["mel"][:, :, frames]
            rows["flow omni"].append((snr_db(mel64[:, :, frames], new_omni), max_err(mel64[:, :, frames], new_omni)))
            rows["flow cosyvoice fp32"].append(
                (snr_db(mel64[:, :, frames], mel32[:, :, frames]), max_err(mel64[:, :, frames], mel32[:, :, frames]))
            )
            rows["flow cosyvoice bf16 autocast"].append(
                (snr_db(mel64[:, :, frames], mel16[:, :, frames]), max_err(mel64[:, :, frames], mel16[:, :, frames]))
            )
            history_omni = new_omni if history_omni is None else torch.cat([history_omni, new_omni], dim=2)
            before = step["speech_offset_before"]
            audio64 = hift_samples(hift64, history_omni, final, device, torch.float64)[:, before:]
            audio32 = hift_samples(hift32, history_omni, final, device, torch.float32)[:, before:]
            delta = step["delta"].reshape(1, -1)
            rows["hift omni"].append((snr_db(audio64, delta), max_err(audio64, delta)))
            rows["hift cosyvoice fp32"].append((snr_db(audio64, audio32), max_err(audio64, audio32)))
            new_chain = mel32[:, :, frames]
            chain_mel = new_chain if chain_mel is None else torch.cat([chain_mel, new_chain], dim=2)
            chain_audio.append(hift_samples(hift32, chain_mel, final, device, torch.float32)[:, before:].cpu())
            omni_audio.append(delta)
        omni_wave = torch.cat(omni_audio, dim=1)
        chain_wave = torch.cat(chain_audio, dim=1)
        chain["numerics audio snr"] = snr_db(chain_wave, omni_wave) if chain_wave.shape == omni_wave.shape else None
        chain["numerics mel snr"] = snr_db(chain_mel, history_omni)
        semantic_mel, semantic_wave = cosyvoice_schedule(cosy, ar_records[request_id], tokens, device)
        chain["semantics mel snr"] = snr_db(semantic_mel, history_omni) if semantic_mel.shape == history_omni.shape else None
        chain["semantics samples omni, cosyvoice"] = (int(omni_wave.shape[1]), int(semantic_wave.shape[1]))
        chain["semantics audio snr"] = (
            snr_db(semantic_wave, omni_wave) if semantic_wave.shape == omni_wave.shape else None
        )
        chains.append(chain)
        print(f"vocoder {request_id} {chain}", flush=True)
    report = {"chains": chains}
    for name, values in rows.items():
        report[name] = {"snr_db": summary([value[0] for value in values]),
                        "max_err": summary([value[1] for value in values])}
    return report


def cosyvoice_schedule(cosy, ar_record: dict, tokens: list[int], device: str):
    """CosyVoice2Model.tts's streaming schedule on a fresh model (cli/model.py:343-376): the real prompt, a
    first hop lengthened by the prompt's pad, hops doubling to 100, then the final over every token, each
    step's mel appended to the HiFT history (token2wav, cli/model.py:425-450)."""
    prompt_token = ar_record["flow_prompt_speech_token"]
    prompt_feat = ar_record["flow_prompt_speech_feat"]
    embedding = ar_record["flow_embedding"]
    pad = math.ceil(prompt_token.shape[1] / HOP) * HOP - prompt_token.shape[1]
    offset, hop, history, audio, emitted = 0, HOP, None, [], 0

    def step(used: list[int], finalize: bool) -> None:
        nonlocal history, emitted
        mel = flow_mel(cosy.model.flow, used, prompt_token, prompt_feat, embedding, finalize, device,
                       torch.float32, False)[:, :, offset * 2 :]
        history = mel if history is None else torch.cat([history, mel], dim=2)
        speech = hift_samples(cosy.model.hift, history, finalize, device, torch.float32)[:, emitted:]
        emitted += speech.shape[1]
        audio.append(speech.cpu())

    while True:
        this_hop = hop + pad if offset == 0 else hop
        if len(tokens) - offset < this_hop + PRE_LOOKAHEAD:
            break
        step(tokens[: offset + this_hop + PRE_LOOKAHEAD], False)
        offset += this_hop
        hop = min(MAX_HOP, hop * 2)
    step(tokens, True)
    return history, torch.cat(audio, dim=1)


def frontend_report(cosy, ar_records: dict) -> dict:
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")[:96]
    rows = []
    by_ids = {tuple(record["text_token_ids"]): request_id for request_id, record in ar_records.items()}
    for sample in samples:
        prompt_text = PROMPT_PREFIX + sample.ref_text
        model_input = cosy.frontend.frontend_zero_shot(sample.target_text, prompt_text, sample.ref_audio, 24000, "")
        ids = tuple(torch.cat([model_input["prompt_text"], model_input["text"]], dim=1)[0].tolist())
        request_id = by_ids.get(ids)
        if request_id is None:
            continue
        record = ar_records[request_id]
        omni_llm = record["llm_prompt_speech_token_ids"]
        cosy_llm = model_input["llm_prompt_speech_token"][0].tolist()
        try:
            normalized = cosy.frontend.text_normalize(sample.target_text, split=True)
        except Exception as error:  # noqa: BLE001 - the normalizer's optional dependencies
            normalized = [f"normalizer unavailable: {type(error).__name__}"]
        rows.append(
            {
                "request": request_id,
                "text_ids_equal": True,
                "llm_prompt_tokens omni, cosyvoice": (len(omni_llm), len(cosy_llm)),
                "llm_prompt_tokens cosyvoice is omni's prefix": omni_llm[: len(cosy_llm)] == cosy_llm,
                "flow_prompt_tokens_equal": torch.equal(
                    record["flow_prompt_speech_token"].reshape(-1).long(),
                    model_input["flow_prompt_speech_token"].reshape(-1).cpu().long(),
                ),
                "prompt_mel_max_err": max_err(model_input["prompt_speech_feat"], record["flow_prompt_speech_feat"]),
                "embedding_max_err": max_err(model_input["flow_embedding"], record["flow_embedding"]),
                "normalized_segments": normalized,
                "normalization_changes_text": normalized != [sample.target_text],
            }
        )
    return {"matched": len(rows), "of_records": len(ar_records), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda"
    from cosyvoice.cli.cosyvoice import CosyVoice3

    root = stages.resolve_checkpoint(MODEL)
    cosy = CosyVoice3(root)
    ar_records = {
        os.path.basename(path)[3:-3]: torch.load(path, weights_only=False)
        for path in sorted(glob.glob(os.path.join(args.records, "ar_*.pt")))
    }
    voc_records = {
        os.path.basename(path)[4:-3]: torch.load(path, weights_only=False)
        for path in sorted(glob.glob(os.path.join(args.records, "voc_*.pt")))
        if os.path.basename(path)[4:-3] in ar_records
    }
    print(f"records ar {len(ar_records)} vocoder {len(voc_records)}", flush=True)
    report = {
        "torch": torch.__version__,
        "tf32 matmul, cudnn": (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32),
        "frontend": frontend_report(cosy, ar_records),
    }
    print("frontend", json.dumps({k: v for k, v in report["frontend"].items() if k != "rows"}), flush=True)
    report["lm"] = lm_report(cosy, root, ar_records, device)
    print("lm", json.dumps(report["lm"]["variants"], indent=1), flush=True)
    report["vocoder"] = vocoder_report(cosy, ar_records, voc_records, device)
    print("vocoder", json.dumps({k: v for k, v in report["vocoder"].items() if k != "chains"}, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
