"""#3132 SIM forensics (10-25 evening): read tokenizer_margins.py's runs and, optionally, the conditioning probe's
records of a served pair.
1. Per run and method, the tokens and references that differ from the float64 reference, and the float64 margins of
   those tokens against the margins of all tokens.
2. Between runs (other processes, other cards), the tokens ONNX Runtime's CUDA session changes, and their margins.
3. With --probe DIR FIRST.jsonl SECOND.jsonl (one or more): per served reference, whether the two servers' speech
   token codes agree, and for those that differ which server's codes equal the float64 reference's, with the smallest
   float64 margin of the reference's tokens (served codes are joined by the probe's digest of the token tensor).
usage: python3 tokenizer_margins_report.py RUN.pt [RUN.pt ...] [--probe DIR FIRST.jsonl SECOND.jsonl ...]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

METHODS = ("ort_cuda", "ort_cpu", "fp32", "tf32")
THRESHOLDS = (1e-5, 1e-4, 1e-3, 1e-2)


def digest(codes):
    return hashlib.sha1(np.asarray([codes], dtype=np.float32).tobytes()).hexdigest()[:16]


def changed(first, second, margins):
    """Float64 margins of the positions where two code lists differ; a length mismatch counts its extra tokens with
    no margin."""
    values = [margins[index] for index, (a, b) in enumerate(zip(first, second)) if a != b]
    return values, abs(len(first) - len(second))


def margin_line(values):
    if not values:
        return "none"
    else:
        ordered = sorted(values)
        return (f"{len(values)} tokens, margin max {ordered[-1]:.2e}, median {ordered[len(ordered) // 2]:.2e}, "
                f"min {ordered[0]:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--probe", nargs="+", default=[])
    args = parser.parse_args()
    runs = {Path(path).stem: torch.load(path, weights_only=False) for path in args.runs}
    first_run = next(iter(runs.values()))["references"]
    all_margins = [margin for reference in first_run.values() for margin in reference["margins"]]
    print(f"{len(first_run)} references, {len(all_margins)} tokens; share of all tokens with a float64 margin below "
          + ", ".join(f"{threshold:.0e}: {np.mean(np.array(all_margins) < threshold):.4%}" for threshold in THRESHOLDS))
    for name, run in runs.items():
        references = run["references"]
        print(f"run {name} (card {run['card']}, {run['versions']}):")
        for method in METHODS:
            values, extra = [], 0
            touched = 0
            for reference in references.values():
                margins, mismatch = changed(reference[method], reference["fp64"], reference["margins"])
                values += margins
                extra += mismatch
                touched += bool(margins or mismatch)
            print(f"  {method:8s} against fp64: {touched} references, {margin_line(values)}, length mismatches {extra}")
    names = list(runs)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            values, extra, touched = [], 0, 0
            for key, reference in runs[first]["references"].items():
                other = runs[second]["references"][key]
                margins, mismatch = changed(reference["ort_cuda"], other["ort_cuda"], reference["margins"])
                values += margins
                extra += mismatch
                touched += bool(margins or mismatch)
            print(f"ort_cuda {first} against {second}: {touched} references, {margin_line(values)}, "
                  f"length mismatches {extra}")
    by_prefix = {key[:16]: reference for key, reference in first_run.items()}
    probe_dir = Path(args.probe[0]) if args.probe else None
    for first_name, second_name in zip(args.probe[1::2], args.probe[2::2]):
        served = []
        for name in (first_name, second_name):
            codes = {}
            for line in (probe_dir / name).read_text().splitlines():
                record = json.loads(line)
                if record["kind"] == "prepared":
                    codes[record["reference"]] = record["llm_prompt_speech_token"]
                else:
                    pass
            served.append(codes)
        counts = {"agree": 0, "first exact": 0, "second exact": 0, "neither exact": 0, "unjoined": 0}
        diverged_margins, agreed_margins = [], []
        for prefix, first_digest in served[0].items():
            reference = by_prefix.get(prefix)
            if reference is None or prefix not in served[1]:
                counts["unjoined"] += 1
                continue
            else:
                smallest = min(reference["margins"])
            exact = digest(reference["fp64"])
            if first_digest == served[1][prefix]:
                counts["agree"] += 1
                agreed_margins.append(smallest)
            else:
                diverged_margins.append(smallest)
                if first_digest == exact:
                    counts["first exact"] += 1
                elif served[1][prefix] == exact:
                    counts["second exact"] += 1
                else:
                    counts["neither exact"] += 1
        print(f"served {first_name} against {second_name}: {counts}")
        print(f"  smallest margin per reference, diverged: {margin_line(diverged_margins)}")
        print(f"  smallest margin per reference, agreed:   {margin_line(agreed_margins)}")


if __name__ == "__main__":
    main()
