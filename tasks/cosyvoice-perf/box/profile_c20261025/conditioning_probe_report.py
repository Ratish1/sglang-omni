"""#3132 SIM forensics (10-25 evening): read conditioning_probe/sitecustomize.py's records of two servers and name, per
request (joined by content key), the first layer whose output differs: the reference conditioning (which field), the
AR's prompt ids, the AR's first token logits, the AR's decode (codes), or none. Then, from the benchmark's audio of the
same requests, how many are byte identical, against how many have identical codes (a request with identical codes
and different audio would be the vocoder's).
usage: python3 conditioning_probe_report.py PROBE_DIR FIRST.jsonl SECOND.jsonl [FIRST_AUDIO_DIR SECOND_AUDIO_DIR]
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

CONDITIONING = ("llm_prompt_speech_token", "flow_prompt_speech_token", "flow_prompt_speech_feat", "flow_embedding")


def load(path):
    prepared, prefill, result = {}, {}, {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if record["kind"] == "prepared":
            prepared[record["rid"]] = record
        elif record["kind"] == "prefill":
            prefill.setdefault(record["rid"], record)
        else:
            result[record["rid"]] = record
    by_key = {}
    for rid, record in prepared.items():
        by_key.setdefault(record["key"], []).append((rid, record, prefill.get(rid), result.get(rid)))
    return by_key


def layer(first, second):
    _, first_prepared, first_prefill, first_result = first
    _, second_prepared, second_prefill, second_result = second
    fields = [field for field in CONDITIONING if first_prepared[field] != second_prepared[field]]
    if fields:
        return "conditioning: " + ", ".join(fields)
    elif first_prepared["input_ids"] != second_prepared["input_ids"]:
        return "prompt ids"
    elif first_prefill is None or second_prefill is None or first_result is None or second_result is None:
        return "incomplete"
    elif (first_prefill["prefix"], first_prefill["extend"]) != (second_prefill["prefix"], second_prefill["extend"]):
        return "radix prefix length"
    elif first_prefill["logits"] != second_prefill["logits"]:
        return "first token logits"
    elif first_result["codes_sha"] != second_result["codes_sha"]:
        return "decode codes"
    else:
        return "same codes"


def main():
    probe_dir, first_name, second_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    first, second = load(probe_dir / first_name), load(probe_dir / second_name)
    keys = sorted(set(first) & set(second))
    layers = collections.Counter()
    references = collections.defaultdict(set)
    for key in keys:
        name = layer(first[key][0], second[key][0])
        layers[name] += 1
        references[name].add(first[key][0][1]["reference"])
    print(f"{first_name} against {second_name}: {len(keys)} requests joined "
          f"({len(first)} and {len(second)} recorded)")
    for name, count in layers.most_common():
        print(f"  {count:5d} requests on {len(references[name]):4d} references: {name}")
    if len(sys.argv) > 5:
        first_audio, second_audio = Path(sys.argv[4]), Path(sys.argv[5])
        same = sum(
            hashlib.sha1((first_audio / wav.name).read_bytes()).digest()
            == hashlib.sha1((second_audio / wav.name).read_bytes()).digest()
            for wav in first_audio.glob("*.wav")
        )
        print(f"  audio byte identical {same} of {len(list(first_audio.glob('*.wav')))}, identical codes "
              f"{layers['same codes']}: a shortfall of audio against codes is the vocoder's")
    else:
        pass


if __name__ == "__main__":
    main()
