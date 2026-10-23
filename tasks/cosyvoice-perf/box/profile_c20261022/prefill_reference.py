"""L5 forensics (10-23): the first token logits prefill_probe recorded from each server (<probe dir>/<tree>/<card>),
each against an fp32 forward of the same prompt embeddings (the checkpoint's weights rounded to the served bf16, every
activation in float32, transformers' eager attention, TF32 off): per server the KL divergence of its softmax from the
reference's, the largest logit error and top-1 agreement over every prompt all servers recorded, split by radix
prefix (none or some); then every pair of servers against each other, and for each pair the prompts where the second
sits farther from the reference than the first.
usage: CUDA_VISIBLE_DEVICES=<card> python3 prefill_reference.py <probe dir> <out json>"""

import itertools
import json
import os
import statistics
import sys
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2Model

from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
PREFIX = "llm.model.model."


def reference_model(device):
    root = resolve_checkpoint(MODEL)
    config = Qwen2Config.from_pretrained(os.path.join(root, "CosyVoice-BlankEN"), attn_implementation="eager")
    model = Qwen2Model(config)
    state = torch.load(os.path.join(root, "llm.pt"), map_location="cpu", weights_only=True)
    backbone = {key[len(PREFIX) :]: value.to(torch.bfloat16).float() for key, value in state.items() if key.startswith(PREFIX)}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    assert not unexpected and all("rotary" in key for key in missing), (missing, unexpected)
    decoder = state["llm_decoder.weight"].to(torch.bfloat16).float()
    return model.to(device).eval(), decoder.to(device)


def kl(reference, logits):
    log_p = torch.log_softmax(reference, dim=-1)
    return float((log_p.exp() * (log_p - torch.log_softmax(logits, dim=-1))).sum())


def summary(values):
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def main():
    probe, out = Path(sys.argv[1]), sys.argv[2]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    servers = sorted(f"{tree.name}/{card.name}" for tree in probe.iterdir() if tree.is_dir() for card in tree.iterdir() if card.is_dir())
    keys = set.intersection(*({path.stem for path in (probe / server).glob("*.pt")} for server in servers))
    device = torch.device("cuda", 0)
    model, decoder = reference_model(device)
    rows = {server: [] for server in servers}
    records = {}
    with torch.inference_mode():
        for key in sorted(keys):
            loaded = {server: torch.load(probe / server / f"{key}.pt", weights_only=True) for server in servers}
            embeds = loaded[servers[0]]["prompt_input_embeds"].float().to(device)
            hidden = model(inputs_embeds=embeds[None], position_ids=torch.arange(embeds.shape[0], device=device)[None])
            reference = (hidden.last_hidden_state[0, -1] @ decoder.T).cpu()
            records[key] = {server: loaded[server]["logits"] for server in servers}
            for server in servers:
                logits = loaded[server]["logits"]
                rows[server].append(
                    {
                        "key": key,
                        "prefix": loaded[server]["prefix_len"] > 0,
                        "kl": kl(reference, logits),
                        "max_abs": float((logits - reference).abs().max()),
                        "top1": bool(logits.argmax() == reference.argmax()),
                    }
                )
    report = {"servers": servers, "prompts": len(keys), "against_reference": {}, "pairs": {}}
    for server in servers:
        entry = {}
        for label, chosen in (("all", rows[server]), ("no prefix", [r for r in rows[server] if not r["prefix"]]), ("prefix", [r for r in rows[server] if r["prefix"]])):
            if chosen:
                entry[label] = {
                    "kl": summary([r["kl"] for r in chosen]),
                    "max_abs": summary([r["max_abs"] for r in chosen]),
                    "top1": sum(r["top1"] for r in chosen) / len(chosen),
                }
            else:
                pass
        report["against_reference"][server] = entry
    for first, second in itertools.combinations(servers, 2):
        mutual = [kl(records[key][first], records[key][second]) for key in sorted(keys)]
        farther = sum(b["kl"] > a["kl"] for a, b in zip(rows[first], rows[second]))
        report["pairs"][f"{first} vs {second}"] = {
            "kl": summary(mutual),
            "identical": sum(torch.equal(records[key][first], records[key][second]) for key in keys),
            "second_farther_from_reference": farther,
            "kl_to_reference_mean_ratio": statistics.mean(b["kl"] for b in rows[second]) / statistics.mean(a["kl"] for a in rows[first]),
        }
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
