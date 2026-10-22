"""Whether the talker's departure from HF at a decode step lies in HF's incremental cache or in the served talker.

Runs only in the isolated reference environment, on one card (the talker alone, about 7 GB):
    PYTHONPATH=<4.57.1 env> CUDA_VISIBLE_DEVICES=<card> python3 talker_cache_check.py --trace <file> [--steps a:b]
    or --trace-dir <dir> [--threshold kl]: for every trace, the decode steps departing from HF's cached pass
HF's talker (bf16 Instruct weights) reads the traced request's prompt rows and decode rows twice: one token at a time
on a DynamicCache, as talker_parity.py does, and as one forward over the whole sequence with no cache. Per step it
prints the KL divergence (temperature 0.9, suppressed ids masked) between the two HF passes and of the served logits
from each, and the served talker hidden against each pass's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerForConditionalGeneration

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
TEMPERATURE = 0.9


def load_talker() -> Qwen3OmniMoeTalkerForConditionalGeneration:
    config = AutoConfig.from_pretrained(MODEL).talker_config
    with torch.device("cuda"):
        talker = Qwen3OmniMoeTalkerForConditionalGeneration._from_config(config, torch_dtype=torch.bfloat16)  # noqa: leading-underscore  # HF's own constructor
    root = Path(snapshot_download(MODEL, allow_patterns=["*.json"]))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith("talker.")})
    state = {}
    for shard in shards:
        path = Path(snapshot_download(MODEL, allow_patterns=[shard])) / shard
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                if key.startswith("talker."):
                    state[key[len("talker."):]] = handle.get_tensor(key)
                else:
                    pass
    missing, unexpected = talker.load_state_dict(state, strict=False)
    print(f"talker loaded: {len(state)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return talker.eval()


def log_probs(logits: torch.Tensor, suppressed: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float().masked_fill(suppressed, float("-inf")) / TEMPERATURE, dim=-1)


def kl(value: torch.Tensor, reference: torch.Tensor) -> float:
    support = torch.isfinite(value)
    return float((value.exp() * (value - reference))[support].sum())


@torch.no_grad()
def departures(talker: Qwen3OmniMoeTalkerForConditionalGeneration, trace: dict, threshold: float) -> list[tuple]:
    """The decode steps where the served logits depart from HF's cached pass by more than threshold KL."""
    vocab = talker.config.text_config.vocab_size
    suppressed = torch.zeros(vocab, dtype=torch.bool, device="cuda")
    suppressed[vocab - 1024:] = True
    suppressed[talker.config.codec_eos_token_id] = False
    prompt_rows = trace["prompt_rows"].cuda().to(torch.bfloat16)
    length = prompt_rows.shape[0]
    positions = trace["mrope_positions"].long().cuda()
    next_position = int(positions.max()) + 1
    cache = DynamicCache()
    output = talker.model(inputs_embeds=prompt_rows[None], position_ids=positions[:, None], past_key_values=cache,
                          use_cache=True)
    found = []
    for step, served_logits in enumerate(trace["logits"]):
        hidden = output.last_hidden_state[0, -1]
        divergence = kl(log_probs(served_logits.cuda(), suppressed), log_probs(talker.codec_head(hidden), suppressed))
        if divergence > threshold and step > 0:
            found.append((step, round(divergence, 3)))
        else:
            pass
        if step >= len(trace["feedback_rows"]):
            break
        else:
            pass
        row = (trace["feedback_rows"][step] + trace["text_rows"][step]).cuda().to(torch.bfloat16)
        output = talker.model(inputs_embeds=row.view(1, 1, -1),
                              position_ids=torch.full((3, 1, 1), next_position + step, device="cuda"),
                              past_key_values=cache, use_cache=True,
                              cache_position=torch.tensor([length + step], device="cuda"))
    return found


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--trace-dir", type=Path, help="every trace: the steps departing from HF by over --threshold")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--steps", default="118:130")
    args = parser.parse_args()
    first, last = (int(bound) for bound in args.steps.split(":"))
    talker = load_talker()
    if args.trace_dir is not None:
        for path in sorted(args.trace_dir.glob("*.pt")):
            trace = torch.load(path, weights_only=False)
            if trace.get("prompt_rows") is not None and trace.get("feedback_rows"):
                print(f"{path.stem[:8]} frames {len(trace['logits'])}: {departures(talker, trace, args.threshold)}",
                      flush=True)
            else:
                pass
        return
    else:
        pass
    trace = torch.load(args.trace, weights_only=False)
    vocab = talker.config.text_config.vocab_size
    suppressed = torch.zeros(vocab, dtype=torch.bool, device="cuda")
    suppressed[vocab - 1024:] = True
    suppressed[talker.config.codec_eos_token_id] = False
    prompt_rows = trace["prompt_rows"].cuda().to(torch.bfloat16)
    length = prompt_rows.shape[0]
    positions = trace["mrope_positions"].long().cuda()
    next_position = int(positions.max()) + 1
    steps = min(last, len(trace["feedback_rows"]))
    decode_rows = torch.stack([trace["feedback_rows"][step] + trace["text_rows"][step] for step in range(steps)])
    decode_rows = decode_rows.cuda().to(torch.bfloat16)
    # one forward over the whole sequence, no cache
    whole_positions = torch.cat(
        [positions, (next_position + torch.arange(steps, device="cuda")).view(1, -1).expand(3, -1)], dim=1)
    whole = talker.model(inputs_embeds=torch.cat([prompt_rows, decode_rows])[None],
                         position_ids=whole_positions[:, None], use_cache=False).last_hidden_state[0]
    whole_hidden = torch.cat([whole[length - 1:length], whole[length:length + steps]])
    # one token at a time on a cache
    cache = DynamicCache()
    output = talker.model(inputs_embeds=prompt_rows[None], position_ids=positions[:, None], past_key_values=cache,
                          use_cache=True)
    cached_hidden = [output.last_hidden_state[0, -1]]
    for step in range(steps):
        position = torch.full((3, 1, 1), next_position + step, device="cuda")
        output = talker.model(inputs_embeds=decode_rows[step].view(1, 1, -1), position_ids=position,
                              past_key_values=cache, use_cache=True,
                              cache_position=torch.tensor([length + step], device="cuda"))
        cached_hidden.append(output.last_hidden_state[0, -1])
    cached_hidden = torch.stack(cached_hidden)
    for step in range(first, min(last, steps + 1)):
        cached_log = log_probs(talker.codec_head(cached_hidden[step]), suppressed)
        whole_log = log_probs(talker.codec_head(whole_hidden[step]), suppressed)
        served_log = log_probs(trace["logits"][step].cuda(), suppressed)
        served_hidden = trace["hidden"][step].cuda().float()

        def hidden_error(reference: torch.Tensor) -> float:
            return float((served_hidden - reference.float()).norm() / reference.float().norm())

        print(f"step {step}: KL cached/whole {kl(cached_log, whole_log):.4f} served/cached "
              f"{kl(served_log, cached_log):.4f} served/whole {kl(served_log, whole_log):.4f} | hidden served/cached "
              f"{hidden_error(cached_hidden[step]):.4f} served/whole {hidden_error(whole_hidden[step]):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
