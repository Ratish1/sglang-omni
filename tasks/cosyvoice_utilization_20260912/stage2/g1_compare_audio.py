#!/usr/bin/env python3
"""G1: is one arm's emitted audio byte identical to another's.

Takes two directories written by g1_stream_identity.sh. Checks that the two runs
asked the same thing (same ordered corpus, same client contract, both complete),
then compares the generated audio file of every sample byte for byte. A slice
with no numeric change passes; anything else names the samples that differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IGNORED_CONFIG = {"output_dir", "model"}


def read_json(path: Path):
    return json.loads(path.read_text())


def audio_digests(run: Path) -> dict[str, tuple[str, int]]:
    directory = run / "seedtts" / "measured" / "audio"
    if not directory.is_dir():
        raise SystemExit(f"{run} holds no measured/audio directory")
    digests = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            raw = path.read_bytes()
            digests[path.name] = (hashlib.sha256(raw).hexdigest(), len(raw))
    if not digests:
        raise SystemExit(f"{directory} is empty")
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    runs = [
        read_json(path / "seedtts" / "experiment.json")
        for path in (args.baseline, args.candidate)
    ]
    for label, path, run in zip(
        ("baseline", "candidate"), (args.baseline, args.candidate), runs, strict=True
    ):
        if run["status"] != "complete":
            raise SystemExit(f"{label} {path} is {run['status']}, not complete")
    if runs[0]["ordered_inputs_sha256"] != runs[1]["ordered_inputs_sha256"]:
        raise SystemExit("the two runs did not send the same ordered corpus")
    contracts = [
        {k: v for k, v in run["client_config"].items() if k not in IGNORED_CONFIG}
        for run in runs
    ]
    if contracts[0] != contracts[1]:
        differing = sorted(k for k in contracts[0] if contracts[0][k] != contracts[1].get(k))
        raise SystemExit("client contracts differ: " + ", ".join(differing))
    if contracts[0].get("seed") is None:
        raise SystemExit("G1 compares seeded runs; neither run set a seed")

    baseline, candidate = (audio_digests(path) for path in (args.baseline, args.candidate))
    if set(baseline) != set(candidate):
        raise SystemExit("the two runs wrote different sample files")
    differing = [name for name in baseline if baseline[name] != candidate[name]]
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_head": (args.baseline / "head.txt").read_text().strip(),
        "candidate_head": (args.candidate / "head.txt").read_text().strip(),
        "ordered_inputs_sha256": runs[0]["ordered_inputs_sha256"],
        "seed": contracts[0]["seed"],
        "samples": len(baseline),
        "identical": len(baseline) - len(differing),
        "differing": {
            name: {"baseline": baseline[name], "candidate": candidate[name]}
            for name in sorted(differing)
        },
        "verdict": "pass" if not differing else "fail",
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    raise SystemExit(0 if not differing else 1)


if __name__ == "__main__":
    main()
