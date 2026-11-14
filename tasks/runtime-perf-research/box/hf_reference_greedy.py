"""Greedy replies of the Hugging Face Qwen3-Omni thinker on inputs dumped from a serving preprocessor.

usage: python3 hf_reference_greedy.py --dumps '<prefix>.*.pt' --out reference.json [--runs 2]

Each dump holds the input ids, attention mask and multimodal tensors one served request received (preproc_hash_probe with
OMNI_PREPROC_DUMP_KEYS), so the reference sees the served bytes; the dump's own record carries their hashes. Every leaf the
thinker's forward accepts is passed; the rest are listed. bf16, temperature 0, 256 new tokens, the talker disabled.
"""

import argparse
import glob
import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def leaves(value, out: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                leaves(item, out)
            else:
                out[key] = item
    else:
        pass


def sha(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()[:24]


def tail_unique(text: str) -> float:
    words = text.split()[-60:]
    return len(set(words)) / max(len(words), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    )
    model.disable_talker()
    thinker = model.thinker
    accepted = set(inspect.signature(thinker.forward).parameters)
    report = {
        "transformers_attn": str(thinker.config._attn_implementation),
        "results": [],
    }
    for path in sorted(glob.glob(args.dumps)):
        dump = torch.load(path, weights_only=False)
        record = dump["record"]
        mm: dict = {}
        leaves(dump["full_mm_inputs"], mm)
        passed = {
            key: value.to("cuda") if isinstance(value, torch.Tensor) else value
            for key, value in mm.items()
            if key in accepted and value is not None
        }
        dropped = sorted(key for key in mm if key not in passed)
        input_ids = dump["input_ids"].reshape(1, -1)
        attention_mask = dump["attention_mask"].reshape(1, -1)
        assert (
            sha(input_ids) == record["key"]
        ), "dumped input ids differ from the hashed ones"
        if "use_audio_in_video" in accepted:
            passed.setdefault("use_audio_in_video", False)
        else:
            pass
        for run in range(args.runs):
            with torch.inference_mode():
                output = thinker.generate(
                    input_ids=input_ids.to("cuda"),
                    attention_mask=attention_mask.to("cuda"),
                    **passed,
                    max_new_tokens=256,
                    do_sample=False,
                )
            new_tokens = output[0, input_ids.shape[1] :].tolist()
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            report["results"].append(
                {
                    "key": record["key"],
                    "prompt_tokens": record["prompt_tokens"],
                    "run": run,
                    "passed": sorted(passed),
                    "dropped": dropped,
                    "completion_tokens": len(new_tokens),
                    "tail_unique": round(tail_unique(text), 2),
                    "first_tokens": new_tokens[:32],
                    "text": text,
                }
            )
            print(
                f"{record['key']} run {run} tok {len(new_tokens)} tail {tail_unique(text):.2f} "
                f"{' '.join(text.split())[:160]}",
                flush=True,
            )
        print(f"  passed {sorted(passed)} dropped {dropped}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
