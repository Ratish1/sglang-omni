"""Two boots' layer dumps, request by request: where the served computation first differs between the boots.

Run: python3 boot_identity_read.py <boot A>/layers <boot B>/layers
Requests are paired by their prompt ids (media rows carry their content hash). For each pair: the composed prefill
input embeddings (text rows are table lookups, media rows the encoder's output), the deepstack features, the M-RoPE
positions and the first-token logits, each byte for byte, and the largest absolute differences where they are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def requests(directory: Path) -> dict[tuple[int, ...], tuple[dict, dict]]:
    paired = {}
    for inputs_path in sorted(directory.glob("*_inputs.pt")):
        logits_path = inputs_path.with_name(inputs_path.name.replace("_inputs.pt", "_logits.pt"))
        inputs = torch.load(inputs_path, weights_only=False)
        if not logits_path.exists():
            continue
        else:
            pass
        paired[tuple(inputs["origin_input_ids"])] = (inputs, torch.load(logits_path, weights_only=False))
    return paired


def largest_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max())


def main() -> None:
    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(MODEL).thinker_config.text_config.vocab_size
    first, second = requests(Path(sys.argv[1])), requests(Path(sys.argv[2]))
    prompts = sorted(
        prompt for prompt in set(first) & set(second) if first[prompt][0]["prefix"] == second[prompt][0]["prefix"]
    )
    counts = {"embeds": 0, "media_rows": 0, "deepstack": 0, "positions": 0, "logits": 0, "argmax": 0}
    rows = []
    for prompt in prompts:
        (inputs_a, logits_a), (inputs_b, logits_b) = first[prompt], second[prompt]
        prefix = inputs_a["prefix"]
        # note (ratish): a chunk re-prefilled after a decode retraction extends past the prompt; its rows there are text.
        chunk_rows = inputs_a["input_embeds"].shape[0]
        prompt_media = torch.tensor(prompt[prefix : prefix + chunk_rows], dtype=torch.long) >= vocab_size
        media = torch.zeros(chunk_rows, dtype=torch.bool)
        media[: prompt_media.numel()] = prompt_media
        embeds_equal = torch.equal(inputs_a["input_embeds"], inputs_b["input_embeds"])
        media_equal = torch.equal(inputs_a["input_embeds"][media], inputs_b["input_embeds"][media])
        deepstack_equal = (inputs_a["deepstack_embeds"] is None and inputs_b["deepstack_embeds"] is None) or (
            inputs_a["deepstack_embeds"] is not None
            and inputs_b["deepstack_embeds"] is not None
            and all(torch.equal(x, y) for x, y in zip(inputs_a["deepstack_embeds"], inputs_b["deepstack_embeds"]))
        )
        positions_equal = torch.equal(inputs_a["mrope_positions"], inputs_b["mrope_positions"])
        logits_equal = torch.equal(logits_a["logits"], logits_b["logits"])
        argmax_equal = int(logits_a["logits"].argmax()) == int(logits_b["logits"].argmax())
        for name, equal in (
            ("embeds", embeds_equal), ("media_rows", media_equal), ("deepstack", deepstack_equal),
            ("positions", positions_equal), ("logits", logits_equal), ("argmax", argmax_equal),
        ):
            counts[name] += equal
        rows.append((
            embeds_equal, logits_equal,
            largest_difference(inputs_a["input_embeds"], inputs_b["input_embeds"]),
            largest_difference(logits_a["logits"], logits_b["logits"]),
            int(media.sum()), len(prompt),
        ))
    print(f"{len(prompts)} requests in both boots with the same cached prefix")
    print("equal byte for byte: " + ", ".join(f"{name} {count}" for name, count in counts.items()))
    embeds_same_logits_differ = sum(1 for row in rows if row[0] and not row[1])
    embeds_differ = sum(1 for row in rows if not row[0])
    print(f"embeddings equal but logits differ: {embeds_same_logits_differ}; embeddings differ: {embeds_differ}")
    for row in sorted(rows, key=lambda row: -row[3])[:8]:
        print(
            f"  embeds equal {row[0]!s:5} logits equal {row[1]!s:5} embeds max diff {row[2]:.4g} logits max diff "
            f"{row[3]:.4g} media rows {row[4]} prompt {row[5]}"
        )


if __name__ == "__main__":
    main()
