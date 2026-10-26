"""Generated codec frames of the official qwen-tts 0.1.1 Base voice clone for parity.py's cases: the steps of
Qwen3TTSModel.generate_voice_clone up to the talker's codes (the vocoder decode skipped, it never changes length), at
the checkpoint's own sampling defaults, language English, ICL or x-vector-only per case, max_new_tokens as given. A
row of max_new_tokens frames never emitted codec EOS.

usage: /data/refenv/bin/python parity_reference.py CASES.jsonl OUT.json --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
       --card 1 [--batch 64] [--max-new-tokens 300] [--seed 0]

Runs in a venv holding qwen-tts 0.1.1's pins (transformers 4.57.3, accelerate 1.12.0, torch 2.10.0): its talker does
not load under the serving image's Transformers.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map=f"cuda:{args.card}", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        groups[case["set"]].append(case)
    records = []
    batch_index = 0
    for set_name, cases in groups.items():
        for start in range(0, len(cases), args.batch):
            batch = cases[start : start + args.batch]
            began = time.perf_counter()
            torch.manual_seed(args.seed + batch_index)
            batch_index += 1
            with torch.no_grad():
                items = model.create_voice_clone_prompt(
                    ref_audio=[case["ref_audio"] for case in batch],
                    ref_text=[case.get("ref_text") for case in batch],
                    x_vector_only_mode=[case["mode"] == "xvec" for case in batch],
                )
                input_ids = model._tokenize_texts([model._build_assistant_text(case["input"]) for case in batch])
                ref_ids = [
                    model._tokenize_texts([model._build_ref_text(item.ref_text)])[0] if item.ref_text else None
                    for item in items
                ]
                codes, _ = model.model.generate(
                    input_ids=input_ids,
                    ref_ids=ref_ids,
                    voice_clone_prompt=model._prompt_items_to_voice_clone_prompt(items),
                    languages=["English"] * len(batch),
                    non_streaming_mode=False,
                    **model._merge_generate_kwargs(max_new_tokens=args.max_new_tokens),
                )
            frames = [int(code.shape[0]) for code in codes]
            records += [{"id": case["id"], "set": case["set"], "frames": count} for case, count in zip(batch, frames)]
            print(
                f"{args.model} {set_name} rows {start}-{start + len(batch) - 1}: "
                f"{sum(count >= args.max_new_tokens - 1 for count in frames)} reached {args.max_new_tokens - 1} "
                f"(no EOS), "
                f"{time.perf_counter() - began:.0f} s",
                flush=True,
            )
    args.out.write_text(
        json.dumps({"model": args.model, "max_new_tokens": args.max_new_tokens, "records": records}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
