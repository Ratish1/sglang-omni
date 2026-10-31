"""Match queue104's talker captures to their pass and SeedTTS sample, list the bad requests, and check whether a bad
request's talker started from the same prompt rows and reply tokens as the same sample's other passes.

A capture belongs to the pass whose window holds its finish time and to the sample whose target text its user turn
carries (the talker's input ids hold the user turn's token ids; the prompt ends with the target text). Bad: WER above 0.5 or audio over 15 s. For each sample, the prompt rows are compared over the longest prefix
of talker input ids two captures share (the user turn and the assistant template; the reply's first text row follows),
as relative L2 per row group: text rows (text projection of token embeddings, exact for equal ids) and multimodal
rows (hidden projection of the thinker's layer-24 hidden, which moves with the thinker's batch).

usage: python3 capture_read.py /data/q104 > read.txt
"""

import collections
import glob
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

AUDIO_TOKEN = 151675
PROMPT_LEAD = "in the same voice and style:"
ROOT = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

windows = {}
for line in Path(ROOT, "passes.txt").read_text().splitlines():
    name, edge, stamp = line.split()
    windows.setdefault(name, {})[edge] = float(stamp)

samples = {}
bad = {}
for name in sorted(windows, key=lambda n: int(n[4:])):
    wer_path = os.path.join(ROOT, name, "seedtts_en", "wer_results.json")
    speed_path = os.path.join(ROOT, name, "seedtts_en", "speed_results.json")
    if not os.path.exists(wer_path):
        print(f"{name}: no wer_results.json")
        continue
    else:
        pass
    speed = {
        r["id"]: r for r in json.loads(Path(speed_path).read_text())["per_request"]
    }
    for row in json.loads(Path(wer_path).read_text())["per_sample"]:
        samples.setdefault(row["target_text"].strip(), set()).add(row["id"])
        reply = speed.get(row["id"], {})
        if row["wer"] > 0.5 or row["audio_duration_s"] > 15.0:
            bad[(name, row["id"])] = dict(
                wer=row["wer"],
                audio=row["audio_duration_s"],
                asr=row["whisper_text"],
                target=row["target_text"],
                reply_tokens=reply.get("completion_tokens"),
            )
        else:
            pass

by_text = {text: next(iter(ids)) for text, ids in samples.items() if len(ids) == 1}
ambiguous = sum(1 for ids in samples.values() if len(ids) > 1)
print(
    f"samples {len(samples)}, target texts shared by more than one sample {ambiguous}"
)

records = collections.defaultdict(dict)
unmatched = 0
for path in glob.glob(os.path.join(ROOT, "capture", "*.pt")):
    record = torch.load(path, map_location="cpu")
    finished = record["finished_at"]
    pass_name = None
    for name, edge in windows.items():
        if edge.get("start", 0) <= finished <= edge.get("end", float("inf")):
            pass_name = name
        else:
            pass
    ids = [t for t in record["prefill_input_ids"].tolist() if t != AUDIO_TOKEN]
    prompt = tokenizer.decode(ids)
    target = prompt.split(PROMPT_LEAD, 1)[-1].split("<|im_end|>", 1)[0].strip()
    sample = by_text.get(target)
    if pass_name is None or sample is None:
        unmatched += 1
        continue
    else:
        pass
    record["path"] = path
    # note (ratish): a warmup request repeats a sample inside the first pass; keep the later one, the timed request.
    kept = records[sample].get(pass_name)
    if kept is None or kept["finished_at"] < finished:
        records[sample][pass_name] = record
    else:
        pass
print(f"captures matched for {len(records)} samples, unmatched {unmatched}")


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def compare(first: dict, second: dict) -> tuple[float, float, int]:
    # note (ratish): the last 9 prefill rows are the assistant tail, whose last row is the reply's first text row.
    ids_a = first["prefill_input_ids"][:-9]
    ids_b = second["prefill_input_ids"][:-9]
    shared = 0
    for x, y in zip(ids_a.tolist(), ids_b.tolist()):
        if x != y:
            break
        else:
            shared += 1
    rows_a = first["prefill_input_embeds"][:shared]
    rows_b = second["prefill_input_embeds"][:shared]
    audio = ids_a[:shared] == AUDIO_TOKEN
    text_l2 = relative_l2(rows_a[~audio], rows_b[~audio]) if (~audio).any() else 0.0
    audio_l2 = relative_l2(rows_a[audio], rows_b[audio]) if audio.any() else 0.0
    return text_l2, audio_l2, shared


good_text = []
good_audio = []
for sample, per_pass in records.items():
    names = sorted(name for name in per_pass if (name, sample) not in bad)
    for left, right in zip(names, names[1:]):
        text_l2, audio_l2, _ = compare(per_pass[left], per_pass[right])
        good_text.append(text_l2)
        good_audio.append(audio_l2)
good_text.sort()
good_audio.sort()


def quantiles(values: list[float]) -> str:
    if not values:
        return "none"
    else:
        pass
    pick = lambda q: values[min(int(q * len(values)), len(values) - 1)]  # noqa: E731
    return f"n {len(values)} median {pick(0.5):.2e} p99 {pick(0.99):.2e} max {values[-1]:.2e}"


print(f"good pairs, prompt text rows: {quantiles(good_text)}")
print(f"good pairs, prompt multimodal rows: {quantiles(good_audio)}")


def audio_variant(record: dict) -> int:
    ids = record["prefill_input_ids"][:-9]
    rows = record["prefill_input_embeds"][: ids.numel()][ids == AUDIO_TOKEN]
    return hash(rows.contiguous().view(torch.int16).numpy().tobytes())


# note (ratish): a sample's passes grouped by bit-equal multimodal rows; a minority variant is one held by fewer passes
# than the sample's largest group.
variant_of = {}
minority_total = 0
minority_bad = 0
majority_total = 0
majority_bad = 0
for sample, per_pass in records.items():
    groups = collections.Counter()
    for name, record in per_pass.items():
        variant_of[(name, sample)] = audio_variant(record)
        groups[variant_of[(name, sample)]] += 1
    largest = max(groups.values())
    for name in per_pass:
        size = groups[variant_of[(name, sample)]]
        variant_of[(name, sample)] = (size, len(per_pass), len(groups))
        if size < largest:
            minority_total += 1
            minority_bad += (name, sample) in bad
        else:
            majority_total += 1
            majority_bad += (name, sample) in bad
print(
    f"multimodal row variants: majority-variant captures {majority_total} with {majority_bad} bad, "
    f"minority-variant captures {minority_total} with {minority_bad} bad"
)
print()
selection = []
for (name, sample), info in sorted(bad.items(), key=lambda item: -item[1]["audio"]):
    per_pass = records.get(sample, {})
    record = per_pass.get(name)
    if record is not None:
        controls = [
            per_pass[other]["path"]
            for other in sorted(per_pass)
            if other != name and (other, sample) not in bad
        ][:2]
        selection.append(
            {
                "pass": name,
                "sample": sample,
                "bad": record["path"],
                "controls": controls,
                **info,
            }
        )
    else:
        pass
    print(
        f"{name} {sample} wer {info['wer']:.2f} audio {info['audio']:.1f} s | {info['target'][:50]} | {info['asr'][:90]}"
    )
    if record is None:
        print("   no capture")
        continue
    else:
        pass
    sizes = record["batch_sizes"].float()
    group, passes, variants = variant_of[(name, sample)]
    print(
        f"   multimodal rows shared with {group} of {passes} passes ({variants} variants); "
        f"frames {record['codes'].shape[0]} finish {record['finished_reason'][40:75]} "
        f"batch size min {int(sizes.min())} median {int(sizes.median())} max {int(sizes.max())} "
        f"reply tokens {len(record['thinker_token_ids'])} seed {record['sampling']['seed']}"
    )
    for other_name in sorted(per_pass):
        if other_name == name:
            continue
        else:
            pass
        other = per_pass[other_name]
        text_l2, audio_l2, shared = compare(record, other)
        same_reply = record["thinker_token_ids"] == other["thinker_token_ids"]
        print(
            f"   vs {other_name}: shared prefix {shared} of {record['prefill_input_ids'].numel()}, "
            f"text rows {text_l2:.2e}, multimodal rows {audio_l2:.2e}, same reply tokens {same_reply}, "
            f"its frames {other['codes'].shape[0]}{' BAD' if (other_name, sample) in bad else ''}"
        )
Path(ROOT, "selection.json").write_text(json.dumps(selection, indent=1))
print(f"\nselection.json: {len(selection)} bad requests with their controls")
