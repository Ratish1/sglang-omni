"""Byte identity of Video-MME replies across passes and boots, beside co-batch and HF reference replies.

usage: python3 videomme_text_identity.py --passes label=<glob> [label=<glob> ...] [--cobatch label=<file> ...]
                                         [--reference <hf_reference.json> --keys 005-3=<key>,...]

The first pass is the reference. For every other pass: questions whose reply text equals the reference's byte for byte,
and the questions that differ with both letters. Co-batch files print, per target and mode, each distinct reply with
how often it came and whether it equals the reference pass's reply. HF reference replies print the same way.
"""

import argparse
import collections
import glob
import json


def replies(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        return {s["question_id"]: s for s in json.load(handle)["per_sample"]}


def short(text: str) -> str:
    return " ".join((text or "").split())[:110]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", nargs="+", required=True)
    parser.add_argument("--cobatch", nargs="*", default=[])
    parser.add_argument("--reference", default="")
    parser.add_argument("--keys", default="")
    args = parser.parse_args()
    passes = []
    for spec in args.passes:
        label, pattern = spec.split("=", 1)
        for path in sorted(glob.glob(pattern)):
            passes.append(
                (f"{label}:{path.split('/bench/')[-1].split('/')[0]}", replies(path))
            )
    ref_label, ref = passes[0]
    print(f"reference pass {ref_label}")
    for label, other in passes[1:]:
        same = [
            q
            for q in ref
            if q in other and other[q]["raw_response"] == ref[q]["raw_response"]
        ]
        differ = [q for q in ref if q in other and q not in same]
        print(
            f"  {label:34s} identical {len(same)}/{len(ref)}  differ {[(q, ref[q]['predicted'], other[q]['predicted']) for q in differ]}"
        )
    for spec in args.cobatch:
        label, path = spec.split("=", 1)
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        print(f"\nco-batch {label}")
        groups = collections.defaultdict(collections.Counter)
        for record in records:
            groups[(record["target"], record["mode"])][record["text"]] += 1
        for (target, mode), texts in groups.items():
            for text, count in texts.items():
                tag = (
                    "= ref c1"
                    if ref.get(target, {}).get("raw_response") == text
                    else "differs"
                )
                print(f"  {target} {mode:5s} x{count} {tag:8s} {short(text)}")
    if args.reference:
        keys = dict(item.split("=") for item in args.keys.split(",") if item)
        by_key = {key: question for question, key in keys.items()}
        with open(args.reference, encoding="utf-8") as handle:
            report = json.load(handle)
        print(f"\nHF reference ({report['transformers_attn']})")
        for result in report["results"]:
            question = by_key.get(result["key"], result["key"])
            tag = (
                "= ref c1"
                if ref.get(question, {}).get("raw_response") == result["text"]
                else "differs"
            )
            print(
                f"  {question} run {result['run']} tok {result['completion_tokens']:3d} tail {result['tail_unique']:.2f} {tag:8s} {short(result['text'])}"
            )
            print(f"      ours: {short(ref.get(question, {}).get('raw_response', ''))}")


if __name__ == "__main__":
    main()
