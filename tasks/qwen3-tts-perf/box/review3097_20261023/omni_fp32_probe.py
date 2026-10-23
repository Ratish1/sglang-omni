"""How close each arm's served Qwen3-Omni audio sits to the fp32 decode of the same talker codes, paired by request.
Reads the code2wav dumps omni_pair.py keeps (codes and the served waveform of every request), pairs requests whose codes
are identical in both arms, decodes those codes once with code2wav in fp32 (the loader the server uses, TF32 off) as
the truth, and reports per pass the SNR in dB of each arm's waveform against it, the paired difference b minus a with
a bootstrap 95 % interval, and how many requests b brings closer.

usage: PYTHONPATH=<tree> python3 omni_fp32_probe.py LABEL_PASS [LABEL_PASS ...] [--card 2]   (under PAIR_ROOT)
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import load_code2wav_model

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/omni3097/pairs"))
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def snr_db(actual: np.ndarray, truth: np.ndarray) -> float:
    noise = float(np.sum((actual.astype(np.float64) - truth) ** 2))
    return math.inf if noise == 0 else 10 * math.log10(float(np.sum(truth**2)) / noise)


def dumps(directory: Path) -> dict[str, dict]:
    by_codes = {}
    for path in sorted(directory.glob("*.pt")):
        dump = torch.load(path, weights_only=False)
        if dump["codes"].numel() == 0:
            continue
        else:
            pass
        digest = hashlib.sha256(dump["codes"].numpy().tobytes()).hexdigest()
        by_codes[digest] = dump
    return by_codes


def bootstrap(differences: list[float]) -> tuple[float, float]:
    generator = random.Random(0)
    means = sorted(
        statistics.fmean(generator.choices(differences, k=len(differences))) for _ in range(2000)
    )
    return means[50], means[1949]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = f"cuda:{args.card}"
    torch.cuda.set_device(args.card)
    model = load_code2wav_model(MODEL, device=device, dtype="float32").eval()
    report = {}
    for label in args.labels:
        arms = {arm: dumps(ROOT / label / arm / "codes") for arm in ("a", "b")}
        common = sorted(set(arms["a"]) & set(arms["b"]))
        rows = []
        with torch.inference_mode():
            for digest in common:
                codes = arms["a"][digest]["codes"].to(device).transpose(0, 1).unsqueeze(0)
                truth = model(codes).reshape(-1).double().cpu().numpy()
                served = {arm: arms[arm][digest]["audio"] for arm in arms}
                length = min(len(truth), *(len(audio) for audio in served.values()))
                rows.append(
                    {
                        "frames": int(codes.shape[-1]),
                        "length_match": all(len(audio) == len(truth) for audio in served.values()),
                        "a": snr_db(served["a"][:length], truth[:length]),
                        "b": snr_db(served["b"][:length], truth[:length]),
                        "b_against_a": snr_db(served["b"][:length], served["a"][:length].astype(np.float64)),
                    }
                )
        differences = [row["b"] - row["a"] for row in rows]
        low, high = bootstrap(differences)
        result = {
            "requests": {arm: len(arms[arm]) for arm in arms},
            "equal_codes": len(rows),
            "length_match": sum(row["length_match"] for row in rows),
            "snr_a_median": statistics.median(row["a"] for row in rows),
            "snr_b_median": statistics.median(row["b"] for row in rows),
            "b_minus_a_mean": statistics.fmean(differences),
            "b_minus_a_median": statistics.median(differences),
            "b_minus_a_95": [low, high],
            "b_closer": sum(difference > 0 for difference in differences),
            "b_against_a_median": statistics.median(row["b_against_a"] for row in rows),
        }
        report[label] = result
        print(label, json.dumps(result), flush=True)
        (ROOT / label / "fp32_probe.json").write_text(json.dumps({"summary": result, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
