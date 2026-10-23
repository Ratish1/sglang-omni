"""Whether two arms served byte-identical Qwen3-Omni audio: pairs the code2wav dumps omni_pair.py keeps by their codes
and counts, per pass, the requests whose codes match and, among those, the ones whose served waveform is the same bytes.

usage: python3 dump_identity.py LABEL_PASS [LABEL_PASS ...]   (under PAIR_ROOT)
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/omni3097/pairs"))


def dumps(directory: Path) -> dict[str, np.ndarray]:
    by_codes = {}
    for path in sorted(directory.glob("*.pt")):
        dump = torch.load(path, weights_only=False)
        if dump["codes"].numel():
            by_codes[hashlib.sha256(dump["codes"].numpy().tobytes()).hexdigest()] = dump["audio"]
        else:
            pass
    return by_codes


def main() -> None:
    for label in sys.argv[1:]:
        arms = {arm: dumps(ROOT / label / arm / "codes") for arm in ("a", "b")}
        common = sorted(set(arms["a"]) & set(arms["b"]))
        identical = sum(np.array_equal(arms["a"][key], arms["b"][key]) for key in common)
        print(label, json.dumps({"requests": {arm: len(arms[arm]) for arm in arms}, "equal_codes": len(common),
                                 "identical_audio": identical}), flush=True)


if __name__ == "__main__":
    main()
