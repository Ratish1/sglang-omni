"""Each seeded request's audio, main against the branch in the same order (order a: main gpu0; order b: main gpu1),
byte for byte, split by whether both arms generated as many completion tokens. The speech tokenizer's codes vary
across processes, so a request whose tokens differ has no byte identical audio to expect.
usage: python3 audio_identity_c20261020.py <pair dir> <cell>"""

import hashlib
import json
import sys
from pathlib import Path


def records(run_dir):
    with open(run_dir / "speed_results.json") as handle:
        return {record["id"]: record for record in json.load(handle)["per_request"] if record["is_success"]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


pair_dir, cell = Path(sys.argv[1]), sys.argv[2]
for order, main_gpu, branch_gpu in (("a", "gpu0", "gpu1"), ("b", "gpu1", "gpu0")):
    main = records(pair_dir / order / main_gpu / cell)
    branch = records(pair_dir / order / branch_gpu / cell)
    counts = {"same token count": [0, 0], "different token count": [0, 0]}
    differing = []
    for request_id in sorted(main.keys() & branch.keys()):
        group = "same token count" if main[request_id]["completion_tokens"] == branch[request_id]["completion_tokens"] else "different token count"
        is_identical = digest(main[request_id]["wav_path"]) == digest(branch[request_id]["wav_path"])
        counts[group][0] += is_identical
        counts[group][1] += 1
        if group == "same token count" and not is_identical:
            differing.append(request_id)
        else:
            pass
    print(f"order {order}: " + "; ".join(f"{group} {identical} of {total} byte identical" for group, (identical, total) in counts.items())
          + f"; same tokens but different audio: {differing[:10]}")
